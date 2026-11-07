import solvers.«10.py.lean.d».MarkingWork
import lang.py.FormalSW.HeapRuntimeCostBuffers
import lang.py.FormalSW.HeapRuntimeCostRanges

/-! Complete runtime costs for the source marking branch. These include scalar
expressions, RHS allocation, argument evaluation and the slice write. -/
namespace solvers.«10».markingRuntime
open FormalSW.PyAst HeapCalls.Runtime
open markingWork
private abbrev program := Solvers.P10Ast.module

private def writeBody : List Stmt :=
  match (sourceBlock[2]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []

theorem source_mark (env : HeapCalls.Env) (heap : Mutable.Heap) (address index : Nat) (bytes : List UInt8)
    (inside : squareIndex index < bytes.length)
    (getIndex : env "i" = some (.scalar (.int index)))
    (getSize : env "size" = some (.scalar (.int bytes.length)))
    (getSieve : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    ∃ finalEnv, BlockContinuesIn .word program env sourceBlock heap
      (6 * writes bytes.length index + 300) finalEnv (heap.set address (.bytearray (mark bytes index))) ∧
      finalEnv "size" = env "size" ∧ finalEnv "sieve" = env "sieve" := by
  let p := oddValue index
  let start := squareIndex index
  let count := ((bytes.length : Int) - start - 1) / p + 1
  let e1 := HeapCalls.setLocal env "p" (.scalar (.int p))
  let e2 := HeapCalls.setLocal e1 "start" (.scalar (.int start))
  let e3 := HeapCalls.setLocal e2 "count" (.scalar (.int count))
  have positive : 0 < p := by unfold p oddValue; omega
  have countPositive : 0 < count := by
    have := Int.ediv_nonneg (by omega : 0 ≤ (bytes.length : Int) - start - 1)
      (by omega : 0 ≤ (p : Int))
    unfold count
    omega
  have first : BlockContinuesIn .word program env [sourceBlock[0]'(by decide)] heap 13 e1 heap := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (av := .int (2 * index : Nat)) (bv := .int 3)
      (ca := 6) (cb := 1) (work := 1)
    · apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
        (av := .int 2) (bv := .int index)
      · exact EvaluatesIn.constant
      · exact EvaluatesIn.lookup getIndex
      · simp [Mutable.binary, scalarBinary]
      · rfl
    · exact EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, p, oddValue]
    · rfl
  have second : BlockContinuesIn .word program e1 [sourceBlock[1]'(by decide)] heap 18 e2 heap := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (av := .int ((p : Int) * p - 3)) (bv := .int 2)
      (ca := 11) (cb := 1) (work := 1)
    · apply EvaluatesIn.binary (ca := 6) (cb := 1) (work := 1)
        (av := .int ((p : Int) * p)) (bv := .int 3)
      · apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
          (av := .int p) (bv := .int p)
        · exact EvaluatesIn.lookup rfl
        · exact EvaluatesIn.lookup rfl
        · rfl
        · rfl
      · exact EvaluatesIn.constant
      · rfl
      · rfl
    · exact EvaluatesIn.constant
    · change (some (.ok (.scalar (.int (((oddValue index : Int) * oddValue index - 3) / 2))), heap) :
        Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
      rw [← squareIndex_cast]
    · rfl
  have third : BlockContinuesIn .word program e2 [writeBody[0]'(by decide)] heap 23 e3 heap := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (av := .int (((bytes.length : Int) - start - 1) / p))
      (bv := .int 1) (ca := 16) (cb := 1) (work := 1)
    · apply EvaluatesIn.binary (av := .int ((bytes.length : Int) - start - 1))
        (bv := .int p) (ca := 11) (cb := 1) (work := 1)
      · apply EvaluatesIn.binary (ca := 6) (cb := 1) (work := 1)
          (av := .int ((bytes.length : Int) - start)) (bv := .int 1)
        · apply EvaluatesIn.binary (ca := 1) (cb := 1) (work := 1)
            (av := .int bytes.length) (bv := .int start)
          · exact EvaluatesIn.lookup (by simpa [e2, e1, HeapCalls.setLocal] using getSize)
          · exact EvaluatesIn.lookup rfl
          · rfl
          · rfl
        · exact EvaluatesIn.constant
        · rfl
        · rfl
      · exact EvaluatesIn.lookup (by simp [e2, e1, HeapCalls.setLocal])
      · simp [Mutable.binary, scalarBinary, Nat.ne_of_gt positive, positive]
      · rfl
    · exact EvaluatesIn.constant
    · rfl
    · rfl
  have fourth : BlockContinuesIn .word program e3 [writeBody[1]'(by decide)] heap
      (6 * count.toNat + 230) e3 (heap.set address (.bytearray (mark bytes index))) := by
    have evaluated : EvaluatesIn .word program e3
        (.binOp (.constant (.bytes [0])) .mult (.name "count")) heap
        (count.toNat + 7) (.scalar (.bytes (List.replicate count.toNat 0))) heap := by
      have run := EvaluatesIn.binary (globals := []) (arithmetic := .word) (module := program)
        (env := e3) (a := .constant (.bytes [0])) (b := .name "count") (op := .mult)
        (heap := heap) (value := .scalar (.bytes (List.replicate count.toNat 0))) (work := count.toNat + 2) EvaluatesIn.constant (EvaluatesIn.lookup rfl)
        (by simp [Mutable.binary, scalarBinary])
        (by simp [PrimitiveWork.binary, scalarNumber, PrimitiveWork.integer])
      exact run.mono (by omega)
    have arguments : ArgumentsIn .word program e3 0
        [.name "sieve", .name "start", .constant .none, .name "p"] heap 14
        [.ref address, .scalar (.int start), .scalar .none, .scalar (.int p)] heap :=
      .cons (EvaluatesIn.lookup (by simpa [e3, e2, e1, HeapCalls.setLocal] using getSieve))
        (.cons (EvaluatesIn.lookup (by simp [e3, e2, HeapCalls.setLocal]))
          (.cons EvaluatesIn.constant (.cons (EvaluatesIn.lookup
            (by simp [e3, e2, e1, HeapCalls.setLocal])) .nil)))
    have write : Mutable.setSlice (.ref address) (.scalar (.int start)) (.scalar .none) (.scalar (.int p))
        (.scalar (.bytes (List.replicate count.toNat 0))) heap =
        some (.ok (.scalar .none), heap.set address (.bytearray (mark bytes index))) := by
      apply Mutable.setSlice_buffer heap address bytes _ _ _ _ _
        { start := some start, step := some p } found
      · change (if (p : Int) = 0 then some (.error ⟨"ValueError", []⟩, heap)
          else some (.ok { start := some (start : Int), stop := none, step := some (p : Int) }, heap) :
            Option (Except ExceptionObject FormalSW.Py.Builtin.Slice.Slice × Mutable.Heap)) = _
        rw [if_neg (by omega)]
      · exact mark_slice bytes index inside
    have price : HeapCalls.BufferWork.slice .word bytes.length (.scalar (.int start)) (.scalar .none)
        (.scalar (.int p)) (List.replicate count.toNat 0) = some (5 * count.toNat + 205) := by
      have notOne : (p : Int) ≠ 1 := by unfold p oddValue; omega
      have nonzero : count.toNat ≠ 0 := by omega
      simp [HeapCalls.BufferWork.slice, HeapCalls.BufferWork.sliceWidth, HeapCalls.BufferWork.bound,
        PrimitiveWork.integer, scalarNumber, notOne, nonzero, Nat.add_mul]
      omega
    exact (BlockContinuesIn.buffer_setSlice (lower := some (.name "start")) (upper := none)
      (stride := some (.name "p")) evaluated arguments found write price).mono (by omega)
  have condition : EvaluatesIn .word program e2 (.compare (.name "start") [(.lt, .name "size")])
      heap 7 (.scalar (.bool true)) heap := by
    apply EvaluatesIn.compare (ca := 1) (cb := 1) (work := 2)
      (EvaluatesIn.lookup rfl)
      (EvaluatesIn.lookup (by simpa [e2, e1, HeapCalls.setLocal] using getSize))
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, start, inside]
    · rfl
  have branch := BlockContinuesIn.branch (yes := writeBody) (no := []) condition (flag := true) rfl
    (third.append fourth)
  refine ⟨e3, (first.append (second.append branch)).mono ?_, rfl, rfl⟩
  have equal : count.toNat = writes bytes.length index := source_count _ _
  omega

end solvers.«10».markingRuntime
