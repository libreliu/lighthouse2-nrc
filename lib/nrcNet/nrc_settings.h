// NRC-specific settings
#define NRC_MAXTRAINPATHLENGTH  5           // Should <= MAXPATHLENGTH
//#define NRC_NUMTRAINRAYS        25
#define NRC_NUMTRAINRAYS       10
#define NRC_TRAINCOMPONENTSIZE  6           // how many float4 one struct component occupies
#define NRC_INPUTDIM  64

#define NRC_ENABLE_DEBUG_VIEW_PRIMARY

//#define NRC_DUMP(X, ...) 
#define NRC_DUMP(X, ...)  printf(X "\n", ##__VA_ARGS__)
#define NRC_DUMP_WARN(X, ...)  printf(X "\n", ##__VA_ARGS__)
#define NRC_DUMP_INFO(X, ...)  printf(X "\n", ##__VA_ARGS__)
