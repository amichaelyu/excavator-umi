// ESP32 GPIO trigger server over Bluetooth LE.
//
// Each output pin simulates a button press in one of two ways:
//   PULL_LOW   open-drain. Released = high impedance (the target's own pull-up
//              holds the line HIGH); pressed = pulled to GND.
//   PUSH_HIGH  push-pull. Released = driven LOW (0V); pressed = driven HIGH (3.3V).
// INPUT_ONLY pins can't be pressed; their live level is reported instead.
// A "trigger" presses a pin for a number of milliseconds and then releases it.
// Pins can also be pressed and released explicitly.
//
// The ESP32 advertises as "gpio" with a Nordic-UART-style GATT service:
//   RX characteristic (write)   newline-terminated JSON commands (see handleMessage)
//   TX characteristic (notify)  newline-terminated JSON replies and state
// Messages longer than one BLE packet are split across several writes or
// notifications; the newline marks the end of each message.

#include <Arduino.h>
#include <ArduinoJson.h>
#include <NimBLEDevice.h>
#include <driver/gpio.h>

// ---- Configuration ---------------------------------------------------------

static const char *DEVICE_NAME = "gpio";

// Nordic UART Service UUIDs, so generic BLE UART apps can talk to it too.
static const char *SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E";
static const char *RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E";  // central -> ESP32
static const char *TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E";  // ESP32 -> central

// Pins, their ESP32 DevKit silkscreen labels and how they behave.
// - GPIO34/35 have no output driver on the ESP32, so they are INPUT_ONLY (and
//   have no internal pull resistors: an unconnected pin floats).
// - GPIO1/3 (TX0/RX0) are the USB serial port. Using them turns off serial
//   logging. TX0 chatters during every boot, and RX0 is wired to the USB
//   chip's TX line, which idles HIGH and fights RX0 while USB is connected.
// - GPIO5 (strapping pin) and GPIO14 (JTAG TMS) are avoided: they toggle
//   briefly during boot, before setup() runs, which a push-high pin turns into
//   a short press on every reset.
// - GPIO2 is avoided: the on-board LED to GND leaks current from the target.
enum PinMode : uint8_t { PULL_LOW, PUSH_HIGH, INPUT_ONLY };

struct PinConfig {
  uint8_t pin;
  const char *name;
  PinMode mode;
};

static const PinConfig PINS[] = {
    {32, "D32", PUSH_HIGH},
    {33, "D33", PUSH_HIGH},
    {25, "D25", PUSH_HIGH},
    {26, "D26", PUSH_HIGH},
    {19, "D19", PUSH_HIGH},
    {21, "D21", PUSH_HIGH},
    {22, "D22", PUSH_HIGH},
    {23, "D23", PUSH_HIGH},
};
static const size_t NUM_PINS = sizeof(PINS) / sizeof(PINS[0]);

static const uint32_t DEFAULT_PULSE_MS = 200;
static const uint32_t MAX_PULSE_MS = 60000;
static const size_t MAX_LINE = 256;  // longest accepted command

// ---- State -----------------------------------------------------------------

enum CmdType : uint8_t { CMD_TRIGGER, CMD_PRESS, CMD_RELEASE, CMD_CLIENT_GONE };

struct Cmd {
  CmdType type;
  int8_t index;       // index into PINS, or -1 for all pins
  uint32_t ms;        // pulse length for CMD_TRIGGER
  uint32_t clientId;  // owner of an auto-release hold, 0 = none
};

struct PinState {
  bool pressed = false;
  int level = LOW;         // last read level, INPUT_ONLY pins

  uint32_t releaseAt = 0;  // millis() deadline for a pulse, 0 = no deadline
  uint32_t holder = 0;     // client holding the pin pressed, 0 = none
};

// Something that happened in the BLE host task, handed to loop() so that all
// pin changes and replies happen on one thread.
struct RxEvent {
  uint16_t conn;
  bool gone;  // the central disconnected
  char text[MAX_LINE];
};

static PinState pinState[NUM_PINS];
static QueueHandle_t rxQueue;
static bool stateDirty = true;

static NimBLECharacteristic *txChar;
static volatile uint16_t peerMtu = 23;  // smallest MTU of the connected centrals

// Connection handles start at 0; client ids must be nonzero ("no holder").
static uint32_t clientIdFor(uint16_t conn) { return (uint32_t)conn + 1; }

static int pinIndex(int pin) {
  for (size_t i = 0; i < NUM_PINS; i++)
    if (PINS[i].pin == pin) return (int)i;
  return -2;
}

static int pinIndexByName(const char *name) {
  for (size_t i = 0; i < NUM_PINS; i++)
    if (strcasecmp(PINS[i].name, name) == 0) return (int)i;
  return -2;
}

// Electrical level that represents "pressed" / "released" for a pin.
static int levelFor(size_t i, bool pressed) {
  bool high = PINS[i].mode == PUSH_HIGH ? pressed : !pressed;
  return high ? HIGH : LOW;
}

static void setPin(size_t i, bool pressed) {
  if (PINS[i].mode == INPUT_ONLY) return;
  digitalWrite(PINS[i].pin, levelFor(i, pressed));
  if (pinState[i].pressed != pressed) stateDirty = true;
  pinState[i].pressed = pressed;
}

static String stateJson(JsonVariantConst id = JsonVariantConst()) {
  JsonDocument doc;
  doc["type"] = "state";
  if (!id.isNull()) doc["id"] = id;
  JsonArray pins = doc["pins"].to<JsonArray>();
  for (size_t i = 0; i < NUM_PINS; i++) {
    JsonObject p = pins.add<JsonObject>();
    p["pin"] = PINS[i].pin;
    p["name"] = PINS[i].name;
    switch (PINS[i].mode) {
      case PUSH_HIGH: p["mode"] = "push-high"; break;
      case PULL_LOW: p["mode"] = "pull-low"; break;
      case INPUT_ONLY: p["mode"] = "input"; p["level"] = pinState[i].level; break;
    }
    p["pressed"] = pinState[i].pressed;
  }
  String out;
  serializeJson(doc, out);
  return out;
}

// Serial logging goes nowhere when TX0/RX0 are used as pins.
class NullPrint : public Print {
 public:
  size_t write(uint8_t) override { return 1; }
};
static NullPrint nullLog;
static Print *logOut = &nullLog;
#define LOG (*logOut)

static bool usesUart0() {
  for (size_t i = 0; i < NUM_PINS; i++)
    if (PINS[i].pin == 1 || PINS[i].pin == 3) return true;
  return false;
}

// ---- BLE transport ---------------------------------------------------------

// Sends one message plus its terminating newline, split into notifications
// that fit the negotiated MTU. conn = BLE_HS_CONN_HANDLE_NONE sends to every
// subscribed central.
static void sendLine(const String &msg, uint16_t conn = BLE_HS_CONN_HANDLE_NONE) {
  if (!txChar || NimBLEDevice::getServer()->getConnectedCount() == 0) return;
  String line = msg + '\n';
  size_t chunk = peerMtu > 3 ? peerMtu - 3 : 20;
  const uint8_t *p = (const uint8_t *)line.c_str();
  for (size_t off = 0; off < line.length(); off += chunk) {
    size_t n = min(chunk, line.length() - off);
    txChar->notify(p + off, n, conn);
  }
}

static void updatePeerMtu() {
  NimBLEServer *srv = NimBLEDevice::getServer();
  uint16_t mtu = 0;
  for (uint16_t h : srv->getPeerDevices()) {
    uint16_t m = srv->getPeerMTU(h);
    if (m && (mtu == 0 || m < mtu)) mtu = m;
  }
  peerMtu = mtu ? mtu : 23;
}

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer *srv, NimBLEConnInfo &info) override {
    LOG.printf("BLE central %u connected from %s\n", info.getConnHandle(),
               info.getAddress().toString().c_str());
    // Short connection interval (7.5-15 ms) keeps command latency low.
    srv->updateConnParams(info.getConnHandle(), 6, 12, 0, 400);
    updatePeerMtu();
  }

  void onDisconnect(NimBLEServer *, NimBLEConnInfo &info, int reason) override {
    LOG.printf("BLE central %u disconnected (reason %d)\n", info.getConnHandle(), reason);
    RxEvent ev{};
    ev.conn = info.getConnHandle();
    ev.gone = true;
    xQueueSend(rxQueue, &ev, portMAX_DELAY);  // must not drop: releases holds
    updatePeerMtu();
  }

  void onMTUChange(uint16_t mtu, NimBLEConnInfo &info) override {
    LOG.printf("BLE central %u MTU %u\n", info.getConnHandle(), mtu);
    updatePeerMtu();
  }
};

// Reassembles newline-terminated commands from RX writes. Commands come from
// one central at a time in practice; a partial line is dropped if a different
// central starts writing.
class RxCallbacks : public NimBLECharacteristicCallbacks {
  RxEvent pending{};
  size_t len = 0;
  bool overflow = false;

  void onWrite(NimBLECharacteristic *c, NimBLEConnInfo &info) override {
    uint16_t conn = info.getConnHandle();
    if (len && pending.conn != conn) {
      len = 0;
      overflow = false;
    }
    pending.conn = conn;
    NimBLEAttValue v = c->getValue();
    for (size_t i = 0; i < v.length(); i++) {
      char ch = (char)v.data()[i];
      if (ch == '\r') continue;
      if (ch != '\n') {
        if (len < MAX_LINE - 1) pending.text[len++] = ch;
        else overflow = true;
        continue;
      }
      if (len && !overflow) {
        pending.text[len] = '\0';
        pending.gone = false;
        if (xQueueSend(rxQueue, &pending, 0) != pdTRUE) LOG.println("rx queue full, command dropped");
      }
      len = 0;
      overflow = false;
    }
  }
};

// ---- Command processing (runs in loop()) -----------------------------------

static void applyToPin(size_t i, const Cmd &c) {
  PinState &s = pinState[i];
  switch (c.type) {
    case CMD_TRIGGER:
      setPin(i, true);
      s.releaseAt = millis() + c.ms;
      if (s.releaseAt == 0) s.releaseAt = 1;
      s.holder = 0;
      break;
    case CMD_PRESS:
      setPin(i, true);
      s.releaseAt = 0;
      s.holder = c.clientId;
      break;
    case CMD_RELEASE:
      setPin(i, false);
      s.releaseAt = 0;
      s.holder = 0;
      break;
    case CMD_CLIENT_GONE:
      if (s.holder != 0 && s.holder == c.clientId) {
        setPin(i, false);
        s.holder = 0;
      }
      break;
  }
}

static void apply(const Cmd &c) {
  if (c.index < 0) {
    for (size_t i = 0; i < NUM_PINS; i++) applyToPin(i, c);
  } else {
    applyToPin((size_t)c.index, c);
  }
}

static void pollInputs() {
  for (size_t i = 0; i < NUM_PINS; i++) {
    if (PINS[i].mode != INPUT_ONLY) continue;
    int level = digitalRead(PINS[i].pin);
    if (level != pinState[i].level) {
      pinState[i].level = level;
      stateDirty = true;
    }
  }
}

static void expirePulses() {
  uint32_t now = millis();
  for (size_t i = 0; i < NUM_PINS; i++) {
    PinState &s = pinState[i];
    if (s.releaseAt && (int32_t)(now - s.releaseAt) >= 0) {
      setPin(i, false);
      s.releaseAt = 0;
    }
  }
}

static void sendError(uint16_t conn, JsonVariantConst id, const char *msg) {
  JsonDocument doc;
  doc["type"] = "error";
  if (!id.isNull()) doc["id"] = id;
  doc["error"] = msg;
  String out;
  serializeJson(doc, out);
  sendLine(out, conn);
}

static void handleMessage(uint16_t conn, const char *data) {
  JsonDocument doc;
  if (deserializeJson(doc, data)) {
    sendError(conn, JsonVariantConst(), "invalid JSON");
    return;
  }
  JsonVariantConst id = doc["id"];
  const char *cmd = doc["cmd"] | "";

  // A state reply carries the request id, so a caller can tell it apart from
  // a broadcast that was already on its way.
  if (strcmp(cmd, "state") == 0) {
    sendLine(stateJson(id), conn);
    return;
  }

  Cmd c{};
  // Only holds sent with "autorelease": true are tied to this connection.
  if (doc["autorelease"] | false) c.clientId = clientIdFor(conn);
  if (strcmp(cmd, "trigger") == 0) c.type = CMD_TRIGGER;
  // "low"/"high" are the original names for press/release, kept for scripts.
  else if (strcmp(cmd, "press") == 0 || strcmp(cmd, "low") == 0) c.type = CMD_PRESS;
  else if (strcmp(cmd, "release") == 0 || strcmp(cmd, "high") == 0) c.type = CMD_RELEASE;
  else {
    sendError(conn, id, "unknown cmd (use trigger, press, release, state)");
    return;
  }

  JsonVariantConst pinVal = doc["pin"];
  if (pinVal.is<const char *>() && strcasecmp(pinVal.as<const char *>(), "all") == 0) {
    c.index = -1;
  } else if (pinVal.is<const char *>()) {
    int idx = pinIndexByName(pinVal.as<const char *>());
    if (idx < 0) {
      sendError(conn, id, "unknown pin name (use D32, D33, D25, D26, D19, D21, D22, D23)");
      return;
    }
    c.index = (int8_t)idx;
  } else if (pinVal.is<int>()) {
    int idx = pinIndex(pinVal.as<int>());
    if (idx < 0) {
      sendError(conn, id, "pin not controllable");
      return;
    }
    c.index = (int8_t)idx;
  } else {
    sendError(conn, id, "missing pin (GPIO number, name like \"D26\", or \"all\")");
    return;
  }

  if (c.index >= 0 && PINS[c.index].mode == INPUT_ONLY) {
    sendError(conn, id, "input-only pin, it can't be pressed");
    return;
  }

  uint32_t ms = doc["ms"] | DEFAULT_PULSE_MS;
  if (ms == 0 || ms > MAX_PULSE_MS) {
    sendError(conn, id, "ms must be 1..60000");
    return;
  }
  c.ms = ms;

  apply(c);

  JsonDocument ack;
  ack["type"] = "ack";
  if (!id.isNull()) ack["id"] = id;
  ack["cmd"] = cmd;
  ack["pin"] = pinVal;
  String out;
  serializeJson(ack, out);
  sendLine(out, conn);
}

static void processEvents() {
  RxEvent ev;
  while (xQueueReceive(rxQueue, &ev, 0) == pdTRUE) {
    if (ev.gone) {
      apply(Cmd{CMD_CLIENT_GONE, -1, 0, clientIdFor(ev.conn)});
    } else {
      handleMessage(ev.conn, ev.text);
    }
  }
}

// ---- Setup / loop ----------------------------------------------------------

void setup() {
  // Put every pin in its released state before anything else runs.
  for (size_t i = 0; i < NUM_PINS; i++) {
    if (PINS[i].mode == INPUT_ONLY) {
      pinMode(PINS[i].pin, INPUT);
      pinState[i].level = digitalRead(PINS[i].pin);
      continue;
    }
    int released = levelFor(i, false);
    gpio_set_level((gpio_num_t)PINS[i].pin, released);
    pinMode(PINS[i].pin, PINS[i].mode == PUSH_HIGH ? OUTPUT : OUTPUT_OPEN_DRAIN);
    digitalWrite(PINS[i].pin, released);
  }

  if (!usesUart0()) {
    Serial.begin(115200);
    logOut = &Serial;
  }
  rxQueue = xQueueCreate(16, sizeof(RxEvent));

  NimBLEDevice::init(DEVICE_NAME);
  NimBLEDevice::setPower(9);  // dBm, the ESP32's maximum
  NimBLEDevice::setMTU(517);

  NimBLEServer *srv = NimBLEDevice::createServer();
  srv->setCallbacks(new ServerCallbacks());
  srv->advertiseOnDisconnect(true);

  NimBLEService *svc = srv->createService(SERVICE_UUID);
  NimBLECharacteristic *rx =
      svc->createCharacteristic(RX_UUID, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
  rx->setCallbacks(new RxCallbacks());
  txChar = svc->createCharacteristic(TX_UUID, NIMBLE_PROPERTY::NOTIFY);
  svc->start();

  NimBLEAdvertising *adv = NimBLEDevice::getAdvertising();
  adv->setName(DEVICE_NAME);
  adv->addServiceUUID(SERVICE_UUID);
  adv->enableScanResponse(true);
  adv->start();

  LOG.printf("BLE advertising as \"%s\" (%s)\n", DEVICE_NAME,
             NimBLEDevice::getAddress().toString().c_str());
}

void loop() {
  processEvents();
  expirePulses();

  static uint32_t lastPoll = 0;
  if (millis() - lastPoll >= 10) {
    lastPoll = millis();
    pollInputs();
  }

  if (stateDirty) {
    stateDirty = false;
    sendLine(stateJson());
  }

  delay(1);  // yield to the RTOS (BLE host task + watchdog)
}
