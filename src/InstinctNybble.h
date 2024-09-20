#define NYBBLE
//number of skills: 47

const int8_t bd[] PROGMEM = { 
30, 0, 0, 1,
  39,  39, -80, -80,  20,  20,  47,  47,
  30,  30, -73, -73,  24,  24,  48,  48,
  26,  26, -64, -64,  25,  25,  46,  46,
  23,  23, -55, -55,  24,  24,  43,  43,
  21,  21, -48, -48,  20,  20,  39,  39,
  21,  21, -41, -41,  15,  15,  32,  32,
  22,  22, -36, -36,   8,   8,  26,  26,
  24,  24, -31, -31,   0,   0,  19,  19,
  26,  26, -28, -28,  -8,  -8,  12,  12,
  29,  29, -25, -25, -15, -15,   4,   4,
  33,  33, -24, -24, -23, -23,  -4,  -4,
  38,  38, -23, -23, -30, -30, -10, -10,
  44,  44, -24, -24, -36, -36, -15, -15,
  51,  51, -26, -26, -41, -41, -17, -17,
  60,  60, -30, -30, -45, -45, -18, -18,
  68,  68, -36, -36, -47, -47, -21, -21,
  77,  77, -40, -40, -47, -47, -25, -25,
  84,  84, -43, -43, -46, -46, -25, -25,
  87,  87, -48, -48, -42, -42, -22, -22,
  89,  89, -53, -53, -37, -37, -18, -18,
  88,  88, -59, -59, -30, -30, -13, -13,
  85,  85, -66, -66, -24, -24,  -5,  -5,
  81,  81, -72, -72, -16, -16,   3,   3,
  77,  77, -78, -78,  -9,  -9,  11,  11,
  71,  71, -82, -82,  -1,  -1,  18,  18,
  64,  64, -86, -86,   5,   5,  25,  25,
  58,  58, -88, -88,  11,  11,  32,  32,
  53,  53, -89, -89,  16,  16,  38,  38,
  48,  48, -87, -87,  18,  18,  43,  43,
  42,  42, -82, -82,  19,  19,  46,  46,
};
const int8_t bk[] PROGMEM = { 
32, 0, 0, 1,
  41,  46, -56, -24,  11,   1, -16, -13,
  39,  57, -57, -23,  11,   3, -18, -15,
  36,  64, -58, -24,  11,  11, -20, -16,
  34,  64, -58, -27,  12,  13, -22, -14,
  31,  62, -59, -30,  12,  18, -25, -13,
  28,  60, -59, -33,  13,  20, -28, -12,
  25,  58, -59, -36,  15,  21, -31, -11,
  22,  57, -61, -38,  16,  19, -30, -10,
  19,  56, -64, -41,  18,  17, -27, -10,
  16,  55, -66, -43,  20,  15, -23, -10,
  13,  53, -66, -45,  23,  13, -18, -11,
  11,  52, -62, -47,  23,  12,  -7, -11,
  13,  50, -52, -50,  19,  11,  -2, -11,
  16,  48, -40, -51,  15,  10,  -2, -12,
  20,  46, -29, -53,  10,  10,  -6, -13,
  35,  44, -26, -54,   3,   9,  -9, -15,
  49,  41, -24, -56,   1,  11, -13, -16,
  59,  39, -23, -57,   4,  11, -16, -18,
  64,  36, -25, -58,  11,  11, -15, -20,
  63,  33, -28, -59,  14,  12, -14, -23,
  62,  31, -30, -59,  18,  12, -13, -26,
  59,  28, -33, -59,  21,  13, -11, -29,
  58,  25, -36, -59,  21,  15, -11, -32,
  57,  22, -39, -61,  18,  16, -10, -30,
  56,  18, -41, -64,  16,  18, -10, -26,
  54,  15, -43, -66,  15,  20, -10, -21,
  53,  12, -45, -65,  13,  23, -11, -16,
  51,  12, -47, -60,  12,  22, -11,  -6,
  49,  14, -50, -51,  11,  17, -11,  -2,
  48,  16, -52, -38,  10,  14, -12,  -2,
  46,  23, -53, -28,  10,   9, -14,  -7,
  43,  38, -55, -25,  10,   2, -15,  -9,
};
const int8_t bkL[] PROGMEM = { 
45, 0, 0, 1,
  26,  37, -56, -58,   6,  -8,  -6,   4,
  25,  47, -58, -55,   6, -10,  -7,   5,
  24,  56, -59, -53,   6, -11,  -8,   5,
  23,  64, -60, -50,   7,  -9, -10,   5,
  22,  66, -61, -48,   8,  -7, -12,   5,
  22,  66, -62, -47,   8,  -4, -13,   5,
  21,  65, -63, -46,   9,   0, -15,   3,
  20,  62, -63, -46,   9,   4, -17,   2,
  19,  59, -64, -45,  10,   7, -19,   0,
  18,  57, -64, -46,  10,   7, -22,   0,
  17,  56, -64, -46,  11,   6, -25,   0,
  16,  54, -64, -47,  12,   5, -28,   0,
  15,  52, -64, -48,  12,   4, -31,   0,
  14,  51, -64, -48,  13,   4, -34,   0,
  13,  49, -63, -49,  14,   3, -38,   0,
  12,  47, -66, -50,  14,   3, -36,   0,
  13,  45, -68, -50,  12,   3, -33,   0,
  13,  43, -72, -51,  11,   3, -26,   0,
  15,  40, -75, -52,   9,   3, -21,   0,
  16,  38, -75, -52,   7,   4, -16,   0,
  20,  36, -75, -53,   5,   5,  -6,   0,
  23,  33, -72, -54,   3,   5,   2,   0,
  27,  31, -67, -54,   1,   6,   7,  -1,
  30,  28, -60, -55,  -1,   7,  10,  -1,
  33,  26, -51, -56,  -2,   8,  11,  -1,
  37,  23, -41, -56,  -3,  10,   9,  -1,
  39,  20, -36, -57,  -4,  12,   7,  -2,
  40,  18, -32, -57,  -3,  13,   5,  -2,
  39,  15, -29, -58,  -3,  15,   1,  -2,
  39,  12, -27, -58,  -1,  17,  -4,  -3,
  38,   9, -27, -59,   0,  20,  -7,  -3,
  37,   6, -29, -59,   1,  22,  -7,  -3,
  37,   3, -31, -60,   1,  25,  -6,  -3,
  36,   0, -34, -60,   1,  28,  -5,  -4,
  35,  -3, -36, -60,   2,  31,  -4,  -4,
  34,  -6, -39, -61,   2,  34,  -4,  -5,
  33, -10, -41, -61,   2,  38,  -4,  -5,
  32, -10, -43, -62,   3,  36,  -3,  -5,
  32, -10, -45, -64,   3,  33,  -3,  -3,
  31,  -6, -47, -64,   3,  26,  -3,  -2,
  30,  -3, -49, -65,   4,  21,  -3,   0,
  29,   1, -51, -64,   4,  16,  -4,   0,
  28,  12, -53, -63,   5,   6,  -4,   2,
  27,  24, -54, -60,   5,  -2,  -5,   3,
  27,  34, -56, -58,   5,  -7,  -6,   4,
};
const int8_t crF[] PROGMEM = { 
42, 0, 10, 1,
  11,  49, -51, -47, -13, -20,   8,   0,
   7,  52, -55, -44, -10, -20,   9,  -1,
   2,  54, -58, -42,  -6, -19,  10,  -3,
  -3,  56, -62, -39,   0, -19,  10,  -4,
  -2,  58, -65, -36,   3, -18,  10,  -5,
   1,  61, -68, -33,   0, -17,   9,  -7,
   4,  63, -71, -31,  -3, -16,   8,  -8,
   7,  67, -74, -28,  -5, -19,   7,  -9,
  11,  67, -76, -25,  -7, -24,   5, -11,
  14,  64, -72, -23,  -9, -25,  -3, -14,
  17,  61, -70, -19, -10, -26,  -2, -16,
  20,  57, -68, -16, -12, -27,  -1, -19,
  23,  53, -66, -12, -14, -27,   0, -22,
  26,  49, -65, -17, -15, -27,   0, -13,
  29,  44, -63, -21, -16, -26,   0,  -8,
  32,  39, -61, -26, -17, -26,   1,  -4,
  35,  35, -59, -30, -18, -24,   2,  -2,
  38,  30, -57, -34, -19, -23,   1,   1,
  41,  26, -54, -39, -19, -21,   1,   4,
  44,  20, -52, -43, -20, -19,   1,   5,
  47,  16, -49, -47, -20, -16,   0,   7,
  49,  11, -47, -51, -20, -13,   0,   8,
  52,   7, -44, -55, -20, -10,  -1,   9,
  54,   2, -42, -58, -19,  -6,  -3,  10,
  56,  -3, -39, -62, -19,   0,  -4,  10,
  58,  -2, -36, -65, -18,   3,  -5,  10,
  61,   1, -33, -68, -17,   0,  -7,   9,
  63,   4, -31, -71, -16,  -3,  -8,   8,
  67,   7, -28, -74, -19,  -5,  -9,   7,
  67,  11, -25, -76, -24,  -7, -11,   5,
  64,  14, -23, -72, -25,  -9, -14,  -3,
  61,  17, -19, -70, -26, -10, -16,  -2,
  57,  20, -16, -68, -27, -12, -19,  -1,
  53,  23, -12, -66, -27, -14, -22,   0,
  49,  26, -17, -65, -27, -15, -13,   0,
  44,  29, -21, -63, -26, -16,  -8,   0,
  39,  32, -26, -61, -26, -17,  -4,   1,
  35,  35, -30, -59, -24, -18,  -2,   2,
  30,  38, -34, -57, -23, -19,   1,   1,
  26,  41, -39, -54, -21, -19,   4,   1,
  20,  44, -43, -52, -19, -20,   5,   1,
  16,  47, -47, -49, -16, -20,   7,   0,
};
const int8_t crL[] PROGMEM = { 
42, 0, 10, 1,
  34,  49, -51, -52, -23, -20,   8,   2,
  33,  52, -55, -52, -23, -20,   9,   2,
  31,  54, -58, -51, -22, -19,  10,   2,
  29,  56, -62, -50, -21, -19,  10,   1,
  29,  58, -65, -49, -19, -18,  10,   1,
  30,  61, -68, -49, -19, -17,   9,   1,
  31,  63, -71, -47, -20, -16,   8,   0,
  33,  67, -74, -47, -21, -19,   7,   0,
  33,  67, -76, -46, -20, -24,   5,   0,
  35,  64, -72, -45, -21, -25,  -3,   0,
  36,  61, -70, -44, -21, -26,  -2,  -1,
  36,  57, -68, -43, -22, -27,  -1,  -1,
  38,  53, -66, -42, -22, -27,   0,  -2,
  39,  49, -65, -43, -22, -27,   0,   1,
  40,  44, -63, -44, -22, -26,   0,   2,
  41,  39, -61, -46, -22, -26,   1,   2,
  42,  35, -59, -47, -22, -24,   2,   3,
  43,  30, -57, -48, -23, -23,   1,   3,
  44,  26, -54, -50, -22, -21,   1,   4,
  45,  20, -52, -51, -23, -19,   1,   4,
  45,  16, -49, -52, -23, -16,   0,   3,
  46,  11, -47, -54, -23, -13,   0,   4,
  48,   7, -44, -54, -23, -10,  -1,   4,
  48,   2, -42, -56, -23,  -6,  -3,   5,
  49,  -3, -39, -57, -23,   0,  -4,   4,
  50,  -2, -36, -58, -23,   3,  -5,   5,
  51,   1, -33, -59, -23,   0,  -7,   5,
  52,   4, -31, -60, -23,  -3,  -8,   5,
  54,   7, -28, -61, -24,  -5,  -9,   5,
  54,  11, -25, -62, -25,  -7, -11,   4,
  53,  14, -23, -61, -26,  -9, -14,   2,
  51,  17, -19, -61, -26, -10, -16,   2,
  50,  20, -16, -60, -26, -12, -19,   2,
  48,  23, -12, -59, -26, -14, -22,   2,
  47,  26, -17, -58, -25, -15, -13,   2,
  45,  29, -21, -58, -25, -16,  -8,   2,
  44,  32, -26, -57, -25, -17,  -4,   2,
  42,  35, -30, -57, -25, -18,  -2,   3,
  41,  38, -34, -56, -25, -19,   1,   2,
  39,  41, -39, -55, -24, -19,   4,   2,
  38,  44, -43, -54, -24, -20,   5,   2,
  36,  47, -47, -53, -23, -20,   7,   2,
};
const int8_t trF[] PROGMEM = { 
33, 0, 0, 1,
  41,  46, -49, -55,  11,   1, -12,  -2,
  44,  35, -47, -61,  11,   3, -10,  -8,
  47,  23, -44, -64,  11,   7,  -9, -18,
  50,  10, -41, -65,  12,  15,  -9, -31,
  51,   3, -38, -61,  15,  22,  -9, -43,
  53,   0, -34, -55,  16,  29, -10, -53,
  55,   0, -30, -53,  19,  31, -11, -55,
  56,   5, -27, -55,  23,  26, -11, -47,
  57,  10, -23, -57,  26,  22, -14, -40,
  57,  14, -19, -57,  30,  19, -16, -34,
  57,  18, -15, -58,  34,  17, -19, -29,
  58,  22, -12, -57,  37,  15, -20, -25,
  61,  26, -14, -56,  31,  13, -15, -22,
  64,  30, -18, -56,  25,  12, -10, -18,
  62,  34, -30, -54,  14,  11,  -3, -16,
  58,  37, -41, -52,   6,  10,   0, -14,
  50,  41, -51, -50,   2,  10,  -1, -12,
  41,  43, -59, -48,   1,  11,  -4, -10,
  29,  46, -64, -45,   4,  11, -12, -10,
  17,  49, -65, -42,  10,  12, -24,  -9,
   6,  50, -62, -39,  20,  14, -40,  -9,
   1,  53, -59, -36,  25,  16, -48,  -9,
  -1,  54, -53, -32,  31,  18, -57, -10,
   3,  56, -55, -28,  29,  21, -50, -12,
   8,  56, -56, -24,  24,  25, -43, -13,
  12,  57, -57, -20,  21,  29, -37, -15,
  17,  57, -57, -16,  17,  33, -32, -18,
  20,  56, -57, -12,  16,  39, -27, -20,
  25,  60, -57, -13,  13,  34, -23, -17,
  29,  64, -56, -17,  12,  26, -20, -10,
  32,  63, -55, -25,  11,  17, -17,  -5,
  36,  60, -53, -37,  10,   8, -15,  -1,
  39,  53, -51, -48,  10,   3, -13,   0,
};
const int8_t trL[] PROGMEM = { 
33, 0, 0, 1,
  49,  46, -49, -56,   7,   1, -12,  -9,
  49,  35, -47, -58,   8,   3, -10, -11,
  50,  23, -44, -60,   9,   7,  -9, -13,
  51,  10, -41, -61,   9,  15,  -9, -16,
  52,   3, -38, -61,  10,  22,  -9, -19,
  52,   0, -34, -60,  10,  29, -10, -21,
  53,   0, -30, -59,  11,  31, -11, -22,
  54,   5, -27, -59,  11,  26, -11, -20,
  54,  10, -23, -59,  12,  22, -14, -19,
  55,  14, -19, -59,  12,  19, -16, -18,
  56,  18, -15, -58,  13,  17, -19, -17,
  56,  22, -12, -58,  14,  15, -20, -16,
  57,  26, -14, -57,  13,  13, -15, -15,
  57,  30, -18, -57,  11,  12, -10, -14,
  56,  34, -30, -56,   8,  11,  -3, -13,
  54,  37, -41, -56,   7,  10,   0, -13,
  51,  41, -51, -55,   6,  10,  -1, -12,
  49,  43, -59, -55,   4,  11,  -4, -10,
  46,  46, -64, -54,   4,  11, -12, -10,
  43,  49, -65, -53,   4,  12, -24,  -9,
  40,  50, -62, -53,   4,  14, -40,  -9,
  38,  53, -59, -52,   5,  16, -48,  -8,
  37,  54, -53, -51,   8,  18, -57,  -8,
  38,  56, -55, -50,   7,  21, -50,  -8,
  39,  56, -56, -49,   7,  25, -43,  -8,
  40,  57, -57, -48,   7,  29, -37,  -7,
  42,  57, -57, -47,   7,  33, -32,  -7,
  42,  56, -57, -46,   7,  39, -27,  -7,
  44,  60, -57, -46,   7,  34, -23,  -6,
  45,  64, -56, -48,   7,  26, -20,  -4,
  46,  63, -55, -50,   7,  17, -17,  -5,
  47,  60, -53, -53,   7,   8, -15,  -5,
  48,  53, -51, -55,   7,   3, -13,  -7,
};
const int8_t vtF[] PROGMEM = { 
14, 0, 0, 1,
  46,  38, -58, -48, -19,  -2,  20,   3,
  39,  38, -49, -48,  -3,  -2,   5,   3,
  38,  38, -48, -48,  -2,  -2,   3,   3,
  38,  40, -48, -50,  -2,  -5,   3,   6,
  38,  47, -48, -59,  -2, -20,   3,  22,
  38,  51, -48, -66,  -2, -33,   3,  34,
  38,  48, -48, -61,  -2, -24,   3,  26,
  38,  42, -48, -53,  -2,  -9,   3,  11,
  38,  38, -48, -48,  -2,  -2,   3,   3,
  38,  38, -48, -48,  -2,  -2,   3,   3,
  42,  38, -52, -48,  -8,  -2,   9,   3,
  48,  38, -60, -48, -23,  -2,  24,   3,
  51,  38, -66, -48, -33,  -2,  34,   3,
  47,  38, -59, -48, -21,  -2,  23,   3,
};
const int8_t vtL[] PROGMEM = { 
36, 0, 0, 1,
  29,  26, -29, -26,  27,  22, -28, -23,
  27,  26, -27, -26,  28,  22, -29, -23,
  26,  25, -26, -25,  30,  24, -31, -25,
  25,  25, -25, -25,  31,  24, -32, -25,
  23,  25, -23, -25,  33,  22, -34, -23,
  20,  23, -20, -23,  35,  21, -36, -22,
  17,  17, -17, -17,  39,  27, -40, -28,
  16,  11, -16, -11,  41,  35, -42, -36,
  13,  10, -13, -10,  42,  38, -43, -39,
  10,  10, -10, -10,  40,  40, -41, -41,
  10,  13, -10, -13,  38,  42, -39, -43,
  11,  16, -11, -16,  35,  41, -36, -42,
  17,  17, -17, -17,  27,  39, -28, -40,
  23,  20, -23, -20,  21,  35, -22, -36,
  25,  23, -25, -23,  22,  33, -23, -34,
  25,  25, -25, -25,  24,  31, -25, -32,
  25,  26, -25, -26,  24,  30, -25, -31,
  26,  27, -26, -27,  22,  28, -23, -29,
  26,  29, -26, -29,  22,  27, -23, -28,
  25,  31, -25, -31,  25,  28, -26, -29,
  23,  33, -23, -33,  27,  30, -28, -31,
  23,  36, -23, -36,  27,  31, -28, -32,
  24,  38, -24, -38,  26,  28, -27, -29,
  29,  41, -29, -41,  20,  24, -21, -25,
  34,  43, -34, -43,  14,  24, -15, -25,
  39,  46, -39, -46,  13,  24, -14, -25,
  42,  46, -42, -46,  16,  24, -17, -25,
  45,  44, -45, -44,  23,  20, -24, -21,
  46,  42, -46, -42,  24,  16, -25, -17,
  44,  37, -44, -37,  24,  12, -25, -13,
  43,  31, -43, -31,  24,  17, -25, -18,
  39,  26, -39, -26,  26,  24, -27, -25,
  37,  24, -37, -24,  30,  26, -31, -27,
  35,  23, -35, -23,  31,  27, -32, -28,
  33,  24, -33, -24,  30,  26, -31, -27,
  30,  26, -30, -26,  27,  24, -28, -25,
};
const int8_t wkF[] PROGMEM = { 
44, 0, 0, 1,
   6,  46, -53, -45,  25,  21,  -6, -19,
   5,  47, -59, -43,  29,  23, -11, -18,
   5,  48, -61, -42,  32,  24, -23, -17,
   7,  49, -61, -41,  30,  26, -33, -16,
  10,  50, -59, -39,  27,  27, -38, -16,
  13,  50, -56, -37,  25,  30, -45, -16,
  16,  50, -52, -35,  23,  32, -50, -16,
  18,  51, -49, -33,  23,  35, -54, -16,
  20,  50, -48, -31,  21,  39, -53, -16,
  23,  51, -50, -29,  20,  39, -50, -17,
  25,  53, -50, -27,  18,  36, -46, -17,
  27,  56, -51, -25,  18,  32, -42, -18,
  29,  59, -51, -22,  18,  26, -39, -19,
  31,  59, -51, -19,  17,  23, -35, -21,
  33,  60, -51, -17,  17,  17, -33, -22,
  35,  55, -51, -16,  17,   9, -30, -21,
  37,  48, -50, -18,  17,   5, -29, -17,
  39,  38, -50, -20,  17,   5, -26, -13,
  41,  27, -49, -23,  17,   8, -24,  -9,
  42,  14, -48, -25,  19,  15, -22,  -8,
  43,  11, -47, -34,  20,  17, -22,  -4,
  45,   9, -46, -45,  20,  20, -19,  -3,
  46,   6, -45, -53,  21,  25, -19,  -6,
  47,   5, -43, -59,  23,  29, -18, -11,
  48,   5, -42, -61,  24,  32, -17, -23,
  49,   7, -41, -61,  26,  30, -16, -33,
  50,  10, -39, -59,  27,  27, -16, -38,
  50,  13, -37, -56,  30,  25, -16, -45,
  50,  16, -35, -52,  32,  23, -16, -50,
  51,  18, -33, -49,  35,  23, -16, -54,
  50,  20, -31, -48,  39,  21, -16, -53,
  51,  23, -29, -50,  39,  20, -17, -50,
  53,  25, -27, -50,  36,  18, -17, -46,
  56,  27, -25, -51,  32,  18, -18, -42,
  59,  29, -22, -51,  26,  18, -19, -39,
  59,  31, -19, -51,  23,  17, -21, -35,
  60,  33, -17, -51,  17,  17, -22, -33,
  55,  35, -16, -51,   9,  17, -21, -30,
  48,  37, -18, -50,   5,  17, -17, -29,
  38,  39, -20, -50,   5,  17, -13, -26,
  27,  41, -23, -49,   8,  17,  -9, -24,
  14,  42, -25, -48,  15,  19,  -8, -22,
  11,  43, -34, -47,  17,  20,  -4, -22,
   9,  45, -45, -46,  20,  20,  -3, -19,
};
const int8_t wkL[] PROGMEM = { 
44, 0, 0, 1,
  36,  46, -53, -50,  11,  21,  -6, -18,
  36,  47, -59, -50,  13,  23, -11, -17,
  35,  48, -61, -50,  14,  24, -23, -16,
  35,  49, -61, -49,  14,  26, -33, -16,
  36,  50, -59, -49,  14,  27, -38, -16,
  37,  50, -56, -48,  13,  30, -45, -16,
  37,  50, -52, -48,  13,  32, -50, -15,
  38,  51, -49, -47,  13,  35, -54, -15,
  39,  50, -48, -47,  13,  39, -53, -15,
  39,  51, -50, -46,  13,  39, -50, -15,
  40,  53, -50, -46,  13,  36, -46, -14,
  41,  56, -51, -46,  13,  32, -42, -14,
  41,  59, -51, -45,  13,  26, -39, -14,
  42,  59, -51, -44,  13,  23, -35, -14,
  43,  60, -51, -43,  13,  17, -33, -14,
  43,  55, -51, -44,  14,   9, -30, -13,
  44,  48, -50, -44,  14,   5, -29, -11,
  44,  38, -50, -45,  15,   5, -26, -11,
  44,  27, -49, -46,  15,   8, -24, -10,
  45,  14, -48, -46,  15,  15, -22,  -9,
  46,  11, -47, -49,  15,  17, -22,  -9,
  46,   9, -46, -51,  15,  20, -19, -10,
  46,   6, -45, -53,  16,  25, -19, -13,
  47,   5, -43, -54,  16,  29, -18, -16,
  47,   5, -42, -56,  16,  32, -17, -18,
  48,   7, -41, -56,  16,  30, -16, -21,
  48,  10, -39, -56,  17,  27, -16, -22,
  49,  13, -37, -56,  17,  25, -16, -23,
  49,  16, -35, -55,  17,  23, -16, -25,
  50,  18, -33, -54,  18,  23, -16, -26,
  50,  20, -31, -53,  20,  21, -16, -26,
  50,  23, -29, -53,  20,  20, -17, -26,
  51,  25, -27, -54,  19,  18, -17, -24,
  51,  27, -25, -54,  18,  18, -18, -23,
  52,  29, -22, -54,  16,  18, -19, -23,
  52,  31, -19, -53,  15,  17, -21, -22,
  52,  33, -17, -53,  13,  17, -22, -22,
  49,  35, -16, -53,  12,  17, -21, -21,
  48,  37, -18, -52,  10,  17, -17, -21,
  45,  39, -20, -52,  10,  17, -13, -20,
  42,  41, -23, -52,   8,  17,  -9, -20,
  39,  42, -25, -51,   9,  19,  -8, -19,
  38,  43, -34, -51,   9,  20,  -4, -19,
  37,  45, -45, -51,  10,  20,  -3, -19,
};

const int8_t balance[] PROGMEM = { 
1, 0, 0, 1,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,};
const int8_t buttUp[] PROGMEM = { 
1, 0, 15, 1,
   20,  40,   0,   0,   5,   5,   3,   3,  90,  90, -45, -45, -60, -60,  -5,  -5,};
const int8_t calib[] PROGMEM = { 
1, 0, 0, 1,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,};
const int8_t dropped[] PROGMEM = { 
1, 0, 75, 1,
    0,  30,   0,   0,  -5,  -5,  15,  15, -75, -75, -60, -60,  60,  60,  30,  30,};
const int8_t lifted[] PROGMEM = { 
1, 0, -75, 1,
    0, -70,   0,   0,   0,   0,   0,   0,  55,  55,  20,  20,  45,  45,   0,   0,};
const int8_t lu[] PROGMEM = { 
1, -30, 15, 1,
  -45,  60, -60,   0,   5,   5,   3,   3, -60,  70, -45, -35,  15, -60,  10, -65,};
const int8_t rest[] PROGMEM = { 
1, 0, 0, 1,
  -30, -80, -45,   0,  -3,  -3,   3,   3,  60,  60, -60, -60, -45, -45,  45,  45,};
const int8_t sit[] PROGMEM = { 
1, 0, -30, 1,
   10, -20, -60,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,};
const int8_t str[] PROGMEM = { 
1, 0, 15, 1,
   10,  70, -30,   0,  -5,  -5,   0,   0, -75, -75, -45, -45,  60,  60, -45, -45,};
const int8_t up[] PROGMEM = { 
1, 0, 0, 1,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,};
const int8_t zero[] PROGMEM = { 
1, 0, 0, 1,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,};

const int8_t ang[] PROGMEM = { 
-5, 0, 0, 1,
 0, 0, 0, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
   43,  53, -98,   0,   0,   0,   0,   0,  60,  60,  -9,  -9,  71,  71, -68, -68,	16,30, 0, 0,
  -76,   1,  59,   0,   0,   0,   0,   0,  62,  60,  -9, -37, -53,  71, -68, -87,	 8, 0, 0, 0,
   33,  73, -70,   0,   0,   0,   0,   0,  62,  60, -34, -37, -53, -65, -81, -87,	 8, 0, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
};
const int8_t bx[] PROGMEM = { 
-9, 0, 0, 1,
 4, 5, 5, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
   10, -20, -94,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	12, 0, 0, 0,
   10, -20, -94,   0,  -5,  -5,  20,  20,  38,  38, -90, -90, 108, 108,  -6,  -6,	12, 0, 0, 0,
    0, -40, -94,   0,  -5,  -5,  20,  20,  45,  45, -49, -49,  -6,  -6, -51, -51,	48, 4, 0, 0,
   16, -42, -94,   0,  -5,  -5,  20,  20, -55,  46, -61, -61, 100, -45, -21, -21,	48, 0, 0, 0,
  -16, -35, -94,   0,  -5,  -5,  20,  20,  38, -65, -61, -61, -45, 100, -21, -21,	48, 0, 0, 0,
    0, -73, -94,   0,  -5,  -5,  20,  20,  46,  46, -49, -49, -26, -26, -40, -40,	16, 2, 0, 0,
    0, -73, -80,   0,  -5,  -5,  20,  20,  46,  46, -78, -78,  39,  40,  46,  46,	12, 0, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	12, 0, 0, 0,
};
const int8_t chr[] PROGMEM = { 
-5, 0, 0, 1,
 2, 3, 3, 
  -45, -11,  60,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	 8, 0, 0, 0,
  -49, -33,  89,   0,  -5,  -5,  20,  20,  31,  30, -84, -66,  81,  60,  45,   7,	 8, 0, 0, 0,
  -49, -17,  89,   0,  -5,  -5,  20,  20, -99,  30, -84, -66,  25,  60,  45,   7,	16, 0, 0, 0,
  -60,   5, 120,   0,  -5,  -5,  20,  20,  43,  30, -84, -66, -39,  60,  45,   7,	48, 0, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	12, 0, 0, 0,
};
const int8_t ck[] PROGMEM = { 
-3, 0, 0, 1,
 0, 1, 2, 
   25,   7, -70,   0,   0,   0,   0,   0,  27,  19, -21, -35,   5,  21, -33,  -5,	12, 1, 0, 0,
  -25,   7,  70,   0,   0,   0,   0,   0,  19,  27, -35, -21,  21,   5,  -5, -33,	 8, 0, 0, 0,
   23,  26, -38,   0,  -2,  -2,   2,   2,  15,  15, -33, -33,  80,  80,   4,   4,	12, 0, 0, 0,
};
const int8_t cmh[] PROGMEM = { 
-22, 0, 0, 1,
 0, 20, 5, 
   46,   6,   0,   0,   0,   0,   0,   0,  29,  26, -29, -26,  27,  22, -28, -23,	 0, 0, 0, 0,
   46,   6,   0,   0,   0,   0,   0,   0,  26,  25, -26, -25,  30,  24, -31, -25,	 0, 0, 0, 0,
   52,   0,   0,   0,   0,   0,   0,   0,  23,  25, -23, -25,  33,  22, -34, -23,	 0, 0, 0, 0,
   52,   0,   0,   0,   0,   0,   0,   0,  17,  17, -17, -17,  39,  27, -40, -28,	 0, 0, 0, 0,
   47,   0,   0,   0,   0,   0,   0,   0,  13,  10, -13, -10,  42,  38, -43, -39,	 0, 0, 0, 0,
   47,   0,   0,   0,   0,   0,   0,   0,  10,  13, -10, -13,  38,  42, -39, -43,	 0, 0, 0, 0,
   52,   0,   0,   0,   0,   0,   0,   0,  17,  17, -17, -17,  27,  39, -28, -40,	 0, 0, 0, 0,
   52,   0,   0,   0,   0,   0,   0,   0,  25,  23, -25, -23,  22,  33, -23, -34,	 0, 0, 0, 0,
   57,   0,   0,   0,   0,   0,   0,   0,  25,  26, -25, -26,  24,  30, -25, -31,	 0, 0, 0, 0,
   57,   0,   0,   0,   0,   0,   0,   0,  26,  29, -26, -29,  22,  27, -23, -28,	 0, 0, 0, 0,
   52,   0,   0,   0,   0,   0,   0,   0,  23,  33, -23, -33,  27,  30, -28, -31,	 0, 0, 0, 0,
   52,   0,   0,   0,   0,   0,   0,   0,  23,  36, -23, -36,  27,  31, -28, -32,	 0, 0, 0, 0,
   59,   0,   0,   0,   0,   0,   0,   0,  29,  41, -29, -41,  20,  24, -21, -25,	 0, 0, 0, 0,
   59,   0,   0,   0,   0,   0,   0,   0,  34,  43, -34, -43,  14,  24, -15, -25,	 0, 0, 0, 0,
   44,   0,   0,   0,   0,   0,   0,   0,  42,  46, -42, -46,  16,  24, -17, -25,	 0, 0, 0, 0,
   44,   0, -33,   0,   0,   0,   0,   0,  45,  44, -45, -44,  23,  20, -24, -21,	 0, 0, 0, 0,
   62,   0, -33,   0,   0,   0,   0,   0,  44,  37, -44, -37,  24,  12, -25, -13,	 0, 0, 0, 0,
   62,   0,   0,   0,   0,   0,   0,   0,  43,  31, -43, -31,  24,  17, -25, -18,	 0, 0, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  37,  24, -37, -24,  30,  26, -31, -27,	 0, 0, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  35,  23, -35, -23,  31,  27, -32, -28,	 0, 0, 0, 0,
   54,   0,   0,   0,   0,   0,   0,   0,  30,  26, -30, -26,  27,  24, -28, -25,	 0, 0, 0, 0,
  -31,  34,  20,   0,   0,   0,   0,   0,  30,  26, -30, -26,  27,  24, -28, -25,	 8, 0, 0, 0,
};
const int8_t dg[] PROGMEM = { 
-11, 0, 0, 1,
 1, 9, 3, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	12, 0, 0, 0,
    0, -32,   0,   0,   0,   0,   0,   0,  62,  62, -47, -47, -20, -20, -30, -30,	32, 0, 0, 0,
   23, -43,  32,   0,   0,   0,   0,   0,  31,  74, -47, -47, -29, -24, -30, -30,	48, 0, 0, 0,
   23, -43,  32,   0,   0,   0,   0,   0,   0,  72, -47, -47,  -1, -18, -30, -30,	64, 0, 0, 0,
    0, -32,   0,   0,   0,   0,   0,   0,  29,  79, -47, -47, -30, -25, -30, -30,	48, 0, 0, 0,
    0, -32,   0,   0,   0,   0,   0,   0,  66,  66, -47, -47, -32, -27, -30, -30,	32, 0, 0, 0,
  -23, -32, -32,   0,   0,   0,   0,   0,  66,  75, -47, -47, -24, -45, -30, -30,	48, 0, 0, 0,
  -23, -45, -32,   0,   0,   0,   0,   0,  66,   6, -47, -47, -24,  -6, -30, -30,	64, 0, 0, 0,
    0, -45,   0,   0,   0,   0,   0,   0,  66,  41, -47, -47, -24, -28, -30, -30,	48, 0, 0, 0,
    0, -24,   0,   0,   0,   0,   0,   0,  62,  62, -47, -47, -20, -20, -30, -30,	32, 0, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -47, -47,  30,  30, -30, -30,	12, 0, 0, 0,
};
const int8_t fiv[] PROGMEM = { 
-8, 0, 0, 1,
 0, 0, 5, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
   10, -20, -94,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	12, 0, 0, 0,
   10, -20, -94,   0,  -5,  -5,  20,  20,  38,  38, -90, -90, 108,  89,  -6,  -6,	12, 2, 0, 0,
    0,  -4, -94,   0,  -5,  -5,  20,  20,  25,  54, -74, -74, -16,  64, -10, -10,	32, 0, 0, 0,
    0,   1, -94,   0,  -5,  -5,  20,  20, -96,  54, -74, -74,  53,  64, -10, -10,	12, 2, 0, 0,
   13, -16, -94,   0,  -5,  -5,  20,  20, -81,  54, -74, -74,  68,  64, -10, -10,	16,20, 0, 0,
   10, -65, -60,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	 8, 0, 0, 0,
   10,  14,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
};
const int8_t gdb[] PROGMEM = { 
-5, 0, -30, 1,
 2, 3, 3, 
    0,  50,   0,   0,   5,   5,  -5,  -5,  35,  35, -25, -25,  -2,  -2, -62, -62,	 8, 0, 0, 0,
    0,  18,  45,   0,  -5,  -5,   5,   5,  36,  12, -59, -35,  40,  84, -12,  24,	 8, 0, 0, 0,
   -8,  38,  54,   0,  -5,  -5,   5,   5,  54,  35, -70, -49,  18,  41,   1,  27,	16, 0, 0, 0,
   -4,  20, -28,   0,  -5,  -5,   5,   5,  25,   0, -56, -34,  71,  78, -10,  15,	48, 0, 0, 0,
   10,  24, -60,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	 8, 0, 0, 0,
};
const int8_t hds[] PROGMEM = { 
-10, 0, 0, 1,
 5, 6, 3, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
    0,  36,   7,   0,   0,   0,   0,   0,  78,  78, -36, -36, -61, -62, -92, -92,	12, 0, 0, 0,
    0, -67,   7,   0,   0,   0,   0,   0,  74,  74, -43, -43, -12, -13, -78, -78,	32, 0, 0, 0,
    0, -77,  -7,   0,   0,   0,   0,   0,   0,   0,  51,  93,  25,  25, -78, -78,	20, 0, 0, 0,
    0, -77,  -7,   0,   0,   0,   0,   0,   0,   0,  51,  93,   9,   9, -78, -78,	16,18, 0, 0,
    0, -77,   7,   0,   0,   0,   0,   0,   0,   0,  93,  51,   9,   9, -78, -78,	16, 0, 0, 0,
    0, -77,  -7,   0,   0,   0,   0,   0,   0,   0,  51,  93,   9,   9, -78, -78,	16, 0, 0, 0,
    0,  49,  -7,   0,   0,   0,   0,   0,  91,  91, -47, -47,  -4,  -4, -78, -78,	12, 0, 0, 0,
    0, -11,   0,   0,   5,   5,   3,   3,  60,  60, -45, -45, -60, -60,  -5,  -5,	12, 0, 0, 0,
    0,  17,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
};
const int8_t hg[] PROGMEM = { 
-9, 0, 0, 1,
 4, 5, 2, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
  -10, -20,  94,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	12, 0, 0, 0,
  -10, -20,  94,   0,  -5,  -5,  20,  20,  38,  38, -61, -61,  97,  97, -26, -26,	12, 0, 0, 0,
  -69, -50, 109,   0,  -5,  -5,  20,  20,  25,  25, -62, -62,  72,  90, -26, -26,	16, 4, 0, 0,
  -19, -36, 110,   0,  -5,  -5,  20,  20, -63, -63, -62, -62,  44,  52, -26, -26,	12, 0, 0, 0,
   33, -21, 110,   0,  -5,  -5,  20,  20, -74, -52, -62, -62,  44,  40, -26, -26,	12, 6, 0, 0,
    0, -28,  84,   0,  -5,  -5,  20,  20,  46,  46, -63, -63,  84,  83, -31, -31,	16, 2, 0, 0,
    0, -28,  70,   0,  -5,  -5,  20,  20,  46,  46, -74, -74,  40,  39,  46,  46,	12,10, 0, 0,
    0,   0,  70,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	12, 0, 0, 0,
};
const int8_t hi[] PROGMEM = { 
-6, 0, 0, 1,
 2, 3, 3, 
    0, -20, -65,   0,   0,   0,   0,   0,  30,  30, -90, -90,  60,  60,  45,  45,	 8, 1, 0, 0,
    0, -20, -65,   0,   0,   0,   0,   0,  30,  50, -90, -74,  60,  43,  45,   5,	 8, 2, 0, 0,
   35, -15, -86,   0,  -3,  -3,   3,   3, -75,  43, -85, -72,  40,  53,  60,   6,	 8, 0, 0, 0,
   40, -10, -64,   0,  -3,  -3,   3,   3, -60,  41, -80, -74,  60,  53,  60,   6,	 4, 0, 0, 0,
    0, -20, -65,   0,   0,   0,   0,   0,  30,  41, -90, -72,  60,  46,  45,   5,	 8, 0, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  41, -30, -30,  30,  30, -30, -30,	12, 0, 0, 0,
};
const int8_t hsk[] PROGMEM = { 
-5, 0, 0, 1,
 2, 3, 3, 
   10, -20, -60,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	 8, 0, 0, 0,
   10, -20, -60,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  26,	 8, 0, 0, 0,
   23, -20, -60,   0,  -5,  -5,  20,  20, -71,  30, -90, -90,  83,  60,  45,  26,	 8, 0, 0, 0,
   21, -27, -60,   0,  -5,  -5,  20,  20, -45,  30, -90, -90,  56,  60,  45,  26,	 8, 0, 0, 0,
   10, -20, -60,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	 8, 0, 0, 0,
};
const int8_t hu[] PROGMEM = { 
-8, 0, 0, 1,
 0, 0, 5, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
   10, -20, -94,   0,  -5,  -5,  20,  20,  30,  30, -72, -72,  60,  60, -25, -25,	12, 0, 0, 0,
   10, -20, -94,   0,  -5,  -5,  20,  20,  38,  38, -61, -61, 108,  89, -25, -25,	12, 2, 0, 0,
    0,  -4, -94,   0,  -5,  -5,  20,  20,  25,  54, -61, -61, -16,  64, -26, -26,	32, 0, 0, 0,
    0,   1, -94,   0,  -5,  -5,  20,  20, -96,  54, -61, -61,  53,  64, -26, -26,	12, 2, 0, 0,
   13, -16, -94,   0,  -5,  -5,  20,  20, -81,  54, -61, -61,  68,  64, -26, -26,	16,20, 0, 0,
   10, -65, -60,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	 8, 0, 0, 0,
   10,  14,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
};
const int8_t kc[] PROGMEM = { 
-9, 0, 0, 1,
 0, 0, 0, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	12, 0, 0, 0,
   22, -33,   0,   0,   0,   0,   0,   0,  52,  52, -20, -36,   0,   0, -61, -43,	16, 0, 0, 0,
   22, -33,   0,   0,   0,   0,   0,   0,  52,  52, -20, -78,   0,   0, -45, -20,	64, 0, 0, 0,
   27, -40,   0,   0,   0,   0,   0,   0,  45,  50, -30, -41,  15,  15, -40, -62,	16, 0, 0, 0,
   27, -40,  90,   0,   0,   0,   0,   0,  59,  37, -30, -48, -60, -10, -33, -62,	32, 0, 0, 0,
   25, -30,  90,   0,   0,   0,   0,   0,  30,  42, -30, -43, -30, -10, -45, -62,	 0, 6, 0, 0,
    5, -18,   0,   0,   0,   0,   0,   0, -45,  42,  -9, -41,  62,  15, -45, -62,	 0, 1, 0, 0,
   -2,  -8,   0,   0,   0,   0,   0,   0, -18,  45,  -9, -43,  60,  15, -45, -40,	16, 0, 0, 0,
   11,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	12, 0, 0, 0,
};
const int8_t nd[] PROGMEM = { 
-5, 0, 0, 1,
 2, 3, 3, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
   36,  40, -83,   0,  -7,  -7,   7,   7,  12,  12, -40, -40, 102, 102, -10, -10,	12, 8, 0, 0,
   17, -44, -83,   0,  -7,  -7,   7,   7,  25,  25, -40, -40,  62,  62, -10, -10,	32, 0, 0, 0,
   17,  40, -70,   0,  -4,  -4,   4,   4,  18,  18, -40, -40,  75,  75, -10, -10,	32, 0, 0, 0,
  -24,  35, -15,   0,  -4,  -4,   4,   4,  12,  12, -40, -40,  84,  84, -10, -10,	 4, 0, 0, 0,
};
const int8_t pd[] PROGMEM = { 
-5, 0, 0, 1,
 3, 4, 2, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
  -84, -73,  65,   0,   0,   0,   0,   0,  73,   0,   0, -73, -60,  88, -88,  60,	12, 4, 0, 0,
   -4, -79,   0,   0,   0,   0,   0,   0,  -3,  16, -16,   3,  90,  56, -56, -90,	 8,20, 0, 0,
   -4, -79,   0,   0,   0,   0,   0,   0,  -3,  16, -16,   3,  90,  56, -56, -73,	32, 0, 0, 0,
   -4, -79,   0,   0,   0,   0,   0,   0,  -3,  15, -15,   3,  90,  60, -60, -90,	32,20, 0, 0,
};
const int8_t pee[] PROGMEM = { 
-5, 0, 0, 1,
 2, 3, 2, 
   30,  20,   0,   0,  15, -10,  60, -10,  40,  40,  85, -45,  10,  60,  15, -45,	 6, 0, 0, 0,
   45,  20, -45,   0,  15, -10,  60, -10,  60,  53,  75, -60, -30,  40,   0, -21,	 2,10, 0, 0,
   30,  20, -55,   0,  15, -10,  60, -10,  40,  40,  85, -45,  10,  50,   0, -45,	 8, 0, 0, 0,
   40,  25, -30,   0,  15, -10,  60, -10,  40,  40,  70, -45,  10,  50, -30, -45,	 8, 0, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
};
const int8_t pu[] PROGMEM = { 
-6, 0, 0, 1,
 1, 2, 3, 
   30,  30,   0,   0,   0,   0,   0,   0,  60,  60,  70,  70,  15,  15, -70, -70,	 6, 0, 0, 0,
    0, -40,   0,   0,   0,   0,   0,   0,  30,  30,  95,  95,  60,  60, -70, -70,	 6, 1, 0, 0,
    5,   5,   0,   0,   0,   0,   0,   0,  75,  75,  55,  55, -50, -50, -75, -75,	 8, 0, 0, 0,
    5,   5,   0,   0,   0,   0,   0,   0,  75, -70,  55,  55, -50,  70, -75, -75,	 8, 0, 0, 0,
   60, -30, -45,   0,   0,   0,   0,   0,  70, -70,  55,   0, -30, -45, -75, -45,	 8, 1, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
};
const int8_t rc[] PROGMEM = { 
-3, 0, 0, 1,
 0, 0, 0, 
   30, -74,  34,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -59, -31,	 8,12, 0, 0,
   62,   0, -20,   0,   0,   0,   0,   0,  -3, -99, 115,   0,  81,   9,  -9, -71,	32,10, 0, 0,
    0,  17,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
};
const int8_t scrh[] PROGMEM = { 
-6, 0, 0, 1,
 3, 4, 5, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
   10, -20, -60,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	 8, 0, 0, 0,
   92, -63, -60,   0,  -5,  -5,  20,  20,  32,  49, -69, -81,  85,  24,  57,  12,	 8, 0, 0, 0,
   70, -63, -71,   0,  -5,  -5,  20,  20,  32,  49, -69, -78,  85,  24,  57, -48,	16, 0, 0, 0,
   66, -67, -52,   0,  -5,  -5,  20,  20,  32,  49, -69, -62,  85,  24,  57, -68,	16, 0, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
};
const int8_t snf[] PROGMEM = { 
-5, 0, 0, 1,
 2, 3, 3, 
   -8, -50, -40,   0,   0,   0,   0,   0,  16,  16, -28, -31,  34,  34, -27, -19,	12, 0, 0, 0,
   -1, -31, -21,   0,   0,   0,   0,   0,  25,  33, -37, -37, -15, -20, -55, -55,	 8, 0, 0, 0,
   -5, -45,  -7,   0,   0,   0,   0,   0,  51,  59, -20, -20, -27, -32, -56, -56,	 4, 0, 0, 0,
    5, -51, -37,   0,   0,   0,   0,   0,  47,  55, -24, -24, -27, -32, -56, -56,	 4, 0, 0, 0,
  -17,   5,  23,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
};
const int8_t stand[] PROGMEM = { 
-6, 0, 0, 1,
 3, 4, 1, 
    0, -20, -60,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	16, 0, 0, 0,
    0, -10,   0,   0,  -5,  -5,  20,  20,  30,  30, -68, -68,  60,  60, -40, -40,	16, 0, 0, 0,
   20, -20, -20,   0,  -5,  -5,  20,  20,  68,  73,   6,   6,  60,  56, -70, -70,	 8, 0, 0, 0,
   20, -20, -20,   0,  -5,  -5,  20,  20, -28,  78,   6,   6,  60,  56, -70, -70,	12, 2, 0, 0,
  -20, -40,  -7,   0,  -5,  -5,  20,  20,  50, -70,   6,   6,  40,  60, -70, -70,	32, 0, 0, 0,
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
};
const int8_t tbl[] PROGMEM = { 
-3, 0, 0, 1,
 0, 1, 2, 
  -33, -38,  40,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
   62,  38, -46,   0,   0,   0,   0,   0,  48,  48, -48, -48,  -6,  -6,   6,   6,	 8, 0, 0, 0,
    0, -75,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  85,  85, -90, -90,	64,20, 0, 0,
};
const int8_t wh[] PROGMEM = { 
-5, 0, 0, 1,
 2, 3, 3, 
    0,   0,   0,   0,   0,   0,   0,   0,  30,  30, -30, -30,  30,  30, -30, -30,	 8, 0, 0, 0,
   17,  14, -47,   0, -13, -13,  13,  13,   6,   6, -54, -54,  72,  72,  12,  12,	 8, 0, 0, 0,
   63,   0, -56,   0, -13, -13,  13,  13,   6,   6, -54, -54,  72,  72,  12,  12,	64, 0, 0, 0,
  -52,   0,  58,   0, -13, -13,  13,  13,   6,   6, -54, -54,  72,  72,  12,  12,	64, 0, 0, 0,
   34,  38,   0,   0,  -5,  -5,   5,   5,  25,  25, -35, -35,  60,  60,   0,   0,	 8, 0, 0, 0,
};
const int8_t wsf[] PROGMEM = { 
-6, 0, 0, 1,
 2, 3, 3, 
   95, -30, -82,   0,  -5,  -5,  20,  20,  30,  30, -90, -90,  60,  60,  45,  45,	12, 2, 0, 0,
   95, -30, -82,   0,  -5,  -5,  20,  20,  30,  43, -90, -71,  85,  44,  45,   0,	 8, 0, 0, 0,
   37, -67, -52,   0,  -5,  -5,  20,  20, -87,  43, -90, -70,  22,  44,  45,   0,	 8, 0, 0, 0,
   49, -28, -70,   0,  -5,  -5,  20,  20, -56,  43, -90, -70, -16,  44,  45,   0,	 4, 4, 0, 0,
  -45, -71, -50,   0,  -5,  -5,  20,  20,  87,  43, -90, -90, -28,  44,  45,   6,	 8, 6, 0, 0,
   64, -39, -60,   0,  -5,  -5,  20,  20,  30,  30, -63, -63,  82,  82,  45,  45,	 8, 0, 0, 0,
};
  const char* skillNameWithType[]={"bdI","bkI","bkLI","crFI","crLI","trFI","trLI","vtFI","vtLI","wkFI","wkLI","balanceI","buttUpI","calibI","droppedI","liftedI","luI","restI","sitI","strI","upI","zeroN","angI","bxI","chrI","ckI","cmhI","dgI","fivI","gdbI","hdsI","hgI","hiI","hskI","huI","kcI","ndI","pdI","peeI","puI","rcI","scrhI","snfI","standI","tblI","whI","wsfI",};
#if !defined(MAIN_SKETCH) || !defined(I2C_EEPROM)
		//if it's not the main sketch to save data or there's no external EEPROM, 
		//the list should always contain all information.
  const int8_t* progmemPointer[] = {bd, bk, bkL, crF, crL, trF, trL, vtF, vtL, wkF, wkL, balance, buttUp, calib, dropped, lifted, lu, rest, sit, str, up, zero, ang, bx, chr, ck, cmh, dg, fiv, gdb, hds, hg, hi, hsk, hu, kc, nd, pd, pee, pu, rc, scrh, snf, stand, tbl, wh, wsf, };
#else	//only need to know the pointers to newbilities, because the intuitions have been saved onto external EEPROM,
	//while the newbilities on progmem are assigned to new addresses
  const int8_t* progmemPointer[] = {zero, };
#endif
//the total byte of instincts is 6999
//the maximal array size is 365 bytes of bkL. 
//Make sure to leave enough memory for SRAM to work properly. Any single skill should be smaller than 400 bytes for safety.