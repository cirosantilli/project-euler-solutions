import solvers.«7.py.lean.d».SetupRuntime

namespace solvers.«7».initialRuntime
open FormalSW.PyAst HeapCalls.Runtime ProjectEulerSolutions.P7
private abbrev program := Solvers.P7Ast.module

private theorem bitLength (n : Nat) (nonzero : n ≠ 0)
    (argument : EvaluatesIn .word program env expression heap cost (.scalar (.int n)) heap) :
    EvaluatesIn .word program env (.builtinCall .intBitLength [expression]) heap
      (cost+4) (.scalar (.int (Nat.log2 n+1 : Nat))) heap := by
  apply EvaluatesIn.builtin_single (work := 1)
    (by intro impossible; cases impossible) (by intro impossible; cases impossible) argument
  · change HeapCalls.scalarResult (some (Math.bitLengthCall (.int n))) heap = _
    simp [HeapCalls.scalarResult, Math.bitLengthCall, FormalSW.Py.Builtin.Int.bitLength, nonzero]
  · rfl

/-- Both bit-length calls and all integer arithmetic in the source estimate. -/
theorem initial_limit (n : Nat) (heap : Mutable.Heap) :
    ReturnsIn .word program "initial_limit" [.scalar (.int n)] heap 50
      (.scalar (.int (initialLimit n))) heap := by
  apply BlockReturnsIn.function (args := [⟨"n", some (.name "int"), none⟩]) (budget := 47) rfl rfl
  by_cases small : n < 6
  · apply BlockReturnsIn.mono (small := 11) _ (by omega)
    apply BlockReturnsIn.branch (cc := 7) (cb := 2) (value := .bool true) (flag := true) (middle := heap)
    · exact EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup rfl) EvaluatesIn.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small]) rfl
    · rfl
    · simp only [initialLimit, if_pos small]
      exact BlockReturnsIn.return_value EvaluatesIn.constant
  · apply BlockReturnsIn.branch (cc := 7) (cb := 38) (value := .bool false) (flag := false) (middle := heap)
    · exact EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup rfl) EvaluatesIn.constant
        (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, small]) rfl
    · rfl
    · apply BlockReturnsIn.assign (ca := 10) (cb := 26) (value := .scalar (.int (Nat.log2 n))) (middle := heap)
      · apply EvaluatesIn.binary (ca := 5) (cb := 1) (work := 1)
          (av := .int (Nat.log2 n+1 : Nat)) (bv := .int 1)
        · exact bitLength n (by omega) (EvaluatesIn.lookup rfl)
        · exact EvaluatesIn.constant
        · simp [Mutable.binary, scalarBinary]
        · rfl
      · apply BlockReturnsIn.return_value (cost := 25)
        apply EvaluatesIn.binary (ca := 20) (cb := 1) (work := 1)
          (av := .int (n * (Nat.log2 n + (Nat.log2 (Nat.log2 n+1)+1)) : Nat)) (bv := .int 10)
        · apply EvaluatesIn.binary (ca := 1) (cb := 15) (work := 1)
            (av := .int n) (bv := .int (Nat.log2 n+(Nat.log2 (Nat.log2 n+1)+1) : Nat))
          · exact EvaluatesIn.lookup (by simp [HeapCalls.setLocal, HeapCalls.argumentEnv])
          · apply EvaluatesIn.binary (ca := 1) (cb := 10) (work := 1)
              (av := .int (Nat.log2 n)) (bv := .int (Nat.log2 (Nat.log2 n+1)+1 : Nat))
            · exact EvaluatesIn.lookup (by simp [HeapCalls.setLocal])
            · apply bitLength (Nat.log2 n+1) (by omega)
              exact EvaluatesIn.binary (work := 1) (av := .int (Nat.log2 n)) (bv := .int 1)
                (EvaluatesIn.lookup (by simp [HeapCalls.setLocal])) EvaluatesIn.constant
                (by simp [Mutable.binary, scalarBinary]) rfl
            · simp [Mutable.binary, scalarBinary]
            · rfl
          · simp [Mutable.binary, scalarBinary]
          · rfl
        · exact EvaluatesIn.constant
        · simp [Mutable.binary, scalarBinary, initialLimit, small, Nat.add_assoc]
        · rfl

end solvers.«7».initialRuntime
