// E5 CAN node: a legitimate periodic ECU, or an attacker (fabrication, masquerade, fuzzing),
// on an MCP2515 + TJA1050 module. The role is chosen at run time over serial, so the same
// image serves the Nucleo-L476RG and the ESP32-S3.
//
// Serial protocol, 115200 baud, one command per line; ids and payloads in hex:
//   BITRATE <125|250|500|1000>                  (re)initialise the controller, kbit/s
//   PERIODIC <id> <period_us> <payload_hex>     add or replace a periodic slot (legit ECU)
//   STOP <id> | STOPALL                         remove a slot / all slots
//   SUSPEND <id> <ms>                           mute a slot for ms (victim side of masquerade)
//   FAB <id> <rate_hz> <ms> <payload_hex|RAND>  fabrication: extra frames for ms
//   FUZZ <rate_hz> <ms>                         random id (0x000-0x7FF) + random payload
//   STAT                                        tx ok/fail counts, TEC/REC, error flags
//   PING                                        -> PONG <micros>, for host clock alignment
//
// Every state change prints "EVT <micros> <what> ..." so the host can label the capture
// (ground truth), the same way ROAD's metadata labels its injection intervals.
// Masquerade = SUSPEND on the victim node + PERIODIC with altered payload on the attacker
// node + STOP when it ends: the frequency stays the same and only the content changes.
#include <Arduino.h>
#include <SPI.h>
#include <mcp2515.h>

#ifndef MCP_CS_PIN
#define MCP_CS_PIN 10
#endif
#ifndef MCP_OSC
#define MCP_OSC MCP_8MHZ
#endif
#ifndef NODE_BOARD
#define NODE_BOARD "unknown"
#endif

static MCP2515 mcp(MCP_CS_PIN);

struct Slot {
  bool used = false;
  uint16_t id = 0;
  uint32_t period_us = 0;
  uint32_t next_us = 0;
  uint32_t mute_until_us = 0;  // SUSPEND
  uint8_t dlc = 0;
  uint8_t data[8] = {0};
};
static const int NSLOTS = 8;
static Slot slots[NSLOTS];

struct Burst {  // FAB and FUZZ
  bool active = false;
  bool fuzz = false;
  bool rand_payload = false;
  uint16_t id = 0;
  uint32_t interval_us = 0, next_us = 0, end_us = 0;
  uint8_t dlc = 0;
  uint8_t data[8] = {0};
  uint32_t sent = 0;
};
static Burst burst;

static uint32_t tx_ok = 0, tx_fail = 0;
static char line[160];
static size_t line_len = 0;

static bool timeReached(uint32_t now, uint32_t t) { return (int32_t)(now - t) >= 0; }

static int parsePayload(const char *hex, uint8_t *out) {
  size_t n = strlen(hex);
  if (n % 2 || n > 16) return -1;
  for (size_t i = 0; i < n / 2; i++) {
    char b[3] = {hex[2 * i], hex[2 * i + 1], 0};
    char *end;
    long v = strtol(b, &end, 16);
    if (*end) return -1;
    out[i] = (uint8_t)v;
  }
  return (int)(n / 2);
}

static bool send(uint16_t id, uint8_t dlc, const uint8_t *data) {
  struct can_frame f;
  f.can_id = id & 0x7FF;
  f.can_dlc = dlc;
  memcpy(f.data, data, dlc);
  if (mcp.sendMessage(&f) == MCP2515::ERROR_OK) { tx_ok++; return true; }
  tx_fail++;
  return false;
}

static bool initBus(int kbps) {
  CAN_SPEED s;
  switch (kbps) {
    case 125: s = CAN_125KBPS; break;
    case 250: s = CAN_250KBPS; break;
    case 500: s = CAN_500KBPS; break;
    case 1000: s = CAN_1000KBPS; break;
    default: return false;
  }
  mcp.reset();
  if (mcp.setBitrate(s, MCP_OSC) != MCP2515::ERROR_OK) return false;
  return mcp.setNormalMode() == MCP2515::ERROR_OK;
}

static Slot *findSlot(uint16_t id, bool create) {
  for (auto &s : slots) if (s.used && s.id == id) return &s;
  if (!create) return nullptr;
  for (auto &s : slots) if (!s.used) { s = Slot(); s.used = true; s.id = id; return &s; }
  return nullptr;
}

static void ok() { Serial.println("OK"); }
static void err(const char *why) { Serial.print("ERR "); Serial.println(why); }

static void handle(char *cmd) {
  char *tok = strtok(cmd, " ");
  if (!tok) return;
  uint32_t now = micros();
  if (!strcmp(tok, "PING")) { Serial.print("PONG "); Serial.println(now); return; }
  if (!strcmp(tok, "BITRATE")) {
    char *a = strtok(nullptr, " ");
    int kbps = a ? atoi(a) : 0;
    if (!initBus(kbps)) return err("bitrate");
    Serial.printf("EVT %lu BITRATE %d\n", (unsigned long)now, kbps);
    return ok();
  }
  if (!strcmp(tok, "PERIODIC")) {
    char *a = strtok(nullptr, " "), *b = strtok(nullptr, " "), *c = strtok(nullptr, " ");
    if (!a || !b || !c) return err("args");
    uint16_t id = strtol(a, nullptr, 16);
    Slot *s = findSlot(id, true);
    if (!s) return err("no free slot");
    int dlc = parsePayload(c, s->data);
    if (dlc < 0) { s->used = false; return err("payload"); }
    s->dlc = dlc;
    s->period_us = strtoul(b, nullptr, 10);
    s->next_us = now;
    Serial.printf("EVT %lu PERIODIC %03X %lu %s\n", (unsigned long)now, id, (unsigned long)s->period_us, c);
    return ok();
  }
  if (!strcmp(tok, "STOP")) {
    char *a = strtok(nullptr, " ");
    if (!a) return err("args");
    uint16_t id = strtol(a, nullptr, 16);
    Slot *s = findSlot(id, false);
    if (!s) return err("no such slot");
    s->used = false;
    Serial.printf("EVT %lu STOP %03X\n", (unsigned long)now, id);
    return ok();
  }
  if (!strcmp(tok, "STOPALL")) {
    for (auto &s : slots) s.used = false;
    burst.active = false;
    Serial.printf("EVT %lu STOPALL\n", (unsigned long)now);
    return ok();
  }
  if (!strcmp(tok, "SUSPEND")) {
    char *a = strtok(nullptr, " "), *b = strtok(nullptr, " ");
    if (!a || !b) return err("args");
    uint16_t id = strtol(a, nullptr, 16);
    Slot *s = findSlot(id, false);
    if (!s) return err("no such slot");
    s->mute_until_us = now + strtoul(b, nullptr, 10) * 1000UL;
    Serial.printf("EVT %lu SUSPEND %03X %s\n", (unsigned long)now, id, b);
    return ok();
  }
  if (!strcmp(tok, "FAB") || !strcmp(tok, "FUZZ")) {
    bool fuzz = !strcmp(tok, "FUZZ");
    burst = Burst();
    burst.fuzz = fuzz;
    char *a = strtok(nullptr, " ");
    if (!fuzz) { if (!a) return err("args"); burst.id = strtol(a, nullptr, 16); a = strtok(nullptr, " "); }
    char *b = strtok(nullptr, " ");
    if (!a || !b) return err("args");
    uint32_t rate = strtoul(a, nullptr, 10);
    if (rate == 0 || rate > 20000) return err("rate");
    burst.interval_us = 1000000UL / rate;
    if (!fuzz) {
      char *c = strtok(nullptr, " ");
      if (!c) return err("args");
      if (!strcmp(c, "RAND")) { burst.rand_payload = true; burst.dlc = 8; }
      else { int d = parsePayload(c, burst.data); if (d < 0) return err("payload"); burst.dlc = d; }
    }
    burst.next_us = now;
    burst.end_us = now + strtoul(b, nullptr, 10) * 1000UL;
    burst.active = true;
    if (fuzz) Serial.printf("EVT %lu FUZZ_START %lu %s\n", (unsigned long)now, (unsigned long)rate, b);
    else Serial.printf("EVT %lu FAB_START %03X %lu %s\n", (unsigned long)now, burst.id, (unsigned long)rate, b);
    return ok();
  }
  if (!strcmp(tok, "STAT")) {
    Serial.printf("STAT board=%s tx_ok=%lu tx_fail=%lu tec=%u rec=%u eflg=0x%02X\n", NODE_BOARD,
                  (unsigned long)tx_ok, (unsigned long)tx_fail, mcp.errorCountTX(), mcp.errorCountRX(),
                  mcp.getErrorFlags());
    return;
  }
  err("unknown command");
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) {}
  SPI.begin();
  bool up = initBus(500);
  randomSeed(micros());
  Serial.printf("READY board=%s cs=%d bus=%s default=500kbps\n", NODE_BOARD, (int)MCP_CS_PIN,
                up ? "up" : "INIT_FAILED");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') { line[line_len] = 0; handle(line); line_len = 0; }
    else if (line_len < sizeof(line) - 1) line[line_len++] = c;
  }
  uint32_t now = micros();
  for (auto &s : slots) {
    if (!s.used || !timeReached(now, s.next_us)) continue;
    s.next_us += s.period_us;
    if (s.mute_until_us && !timeReached(now, s.mute_until_us)) continue;
    if (s.mute_until_us) {
      s.mute_until_us = 0;
      Serial.printf("EVT %lu RESUME %03X\n", (unsigned long)now, s.id);
    }
    send(s.id, s.dlc, s.data);
  }
  if (burst.active) {
    if (timeReached(now, burst.end_us)) {
      burst.active = false;
      Serial.printf("EVT %lu %s_END sent=%lu\n", (unsigned long)now, burst.fuzz ? "FUZZ" : "FAB",
                    (unsigned long)burst.sent);
    } else if (timeReached(now, burst.next_us)) {
      burst.next_us += burst.interval_us;
      uint16_t id = burst.fuzz ? (uint16_t)random(0, 0x800) : burst.id;
      if (burst.fuzz || burst.rand_payload) {
        burst.dlc = 8;
        for (int i = 0; i < 8; i++) burst.data[i] = (uint8_t)random(0, 256);
      }
      if (send(id, burst.dlc, burst.data)) burst.sent++;
    }
  }
}
