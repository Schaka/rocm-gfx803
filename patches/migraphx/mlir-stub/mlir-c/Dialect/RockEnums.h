// Stub for the one rocMLIR header MIGraphX includes unconditionally.
//
// rocMLIR has no gfx8 support, so it is stripped from requirements.txt, but
// src/targets/gpu/mlir.cpp still includes <mlir-c/Dialect/RockEnums.h> whether
// MLIR is enabled or not. Vendoring the two enums it needs is cheaper than
// building all of rocMLIR and LLVM for them.
#ifndef MLIR_C_DIALECT_ROCK_ENUMS_H
#define MLIR_C_DIALECT_ROCK_ENUMS_H

#ifdef __cplusplus
extern "C" {
#endif

enum RocmlirTuningParamSetKind {
  RocmlirTuningParamSetKindQuick = 0,
  RocmlirTuningParamSetKindFull = 1,
  RocmlirTuningParamSetKindExhaustive = 2
};
typedef enum RocmlirTuningParamSetKind RocmlirTuningParamSetKind;

enum RocmlirSplitKSelectionLikelihood { never = 0, maybe = 1, always = 2 };
typedef enum RocmlirSplitKSelectionLikelihood RocmlirSplitKSelectionLikelihood;

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_ROCK_ENUMS_H
