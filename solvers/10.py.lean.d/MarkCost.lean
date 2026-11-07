import solvers.«10.py.lean.d».Source
import ProjectEulerSolutions.ComplexityDrafts.BufferCostCallsAlloc
namespace solvers.«10».complexity
open FormalSW.PyAst
namespace C := FormalSW.PyAst.HeapCalls.BufferCost.Cost
private abbrev program := Solvers.P10Ast.module
set_option maxRecDepth 4096
set_option maxHeartbeats 1500000
private def solveBody : List Stmt :=
  match findFunctionIn program.body "solve" with
  | some (_, body) => executableBody body
  | none => []

private def sieveIteration : List Stmt :=
  match (solveBody[6]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

private def markBody : List Stmt :=
  match sieveIteration with
  | [.ifThen _ body _] => body
  | _ => []

private def writeBody : List Stmt :=
  match (markBody[2]? : Option Stmt) with
  | some (.ifThen _ body _) => body
  | _ => []

private def markEnv (env : HeapCalls.Env) (index size : Nat) : HeapCalls.Env :=
  HeapCalls.setLocal (HeapCalls.setLocal (HeapCalls.setLocal env
    "p" (.scalar (.int (oddValue index)))) "start" (.scalar (.int (squareIndex index))))
    "count" (.scalar (.int (((size : Int) - squareIndex index - 1) / oddValue index + 1)))

/-- The actual assignments and extended-slice write preserve the sieve's address. -/
theorem mark_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (address index : Nat) (bytes : List UInt8)
    (inside : squareIndex index < bytes.length)
    (getIndex : env "i" = some (.scalar (.int index)))
    (getSize : env "size" = some (.scalar (.int bytes.length)))
    (getSieve : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    C.BlockContinuesIn program env markBody heap (69 + 2 * (((bytes.length : Int) - squareIndex index - 1) / oddValue index + 1).toNat) (markEnv env index bytes.length)
      (heap.set address (.bytearray (mark bytes index))) := by
  let p := oddValue index
  let start := squareIndex index
  let count := ((bytes.length : Int) - start - 1) / p + 1
  let e1 := HeapCalls.setLocal env "p" (.scalar (.int p))
  let e2 := HeapCalls.setLocal e1 "start" (.scalar (.int start))
  let e3 := HeapCalls.setLocal e2 "count" (.scalar (.int count))
  have positive : 0 < p := by unfold p oddValue; omega
  have first : C.BlockContinuesIn program env [markBody[0]'(by decide)] heap 11 e1 heap := by
    apply C.BlockContinuesIn.assign
    apply C.EvaluatesIn.binary (av := .scalar (.int (2 * index : Nat)))
      (bv := .scalar (.int 3)) (middle := heap) (afterRight := heap)
      (C.EvaluatesIn.binary C.EvaluatesIn.constant (C.EvaluatesIn.lookup getIndex)
        (by simp [Mutable.binary, scalarBinary])) C.EvaluatesIn.constant
    simp [Mutable.binary, scalarBinary, p, oddValue]
  have second : C.BlockContinuesIn program e1 [markBody[1]'(by decide)] heap 15 e2 heap := by
    apply C.BlockContinuesIn.assign
    apply C.EvaluatesIn.binary (av := .scalar (.int ((p : Int) * p - 3)))
      (bv := .scalar (.int 2)) (middle := heap) (afterRight := heap)
    · exact C.EvaluatesIn.binary
        (C.EvaluatesIn.binary (C.EvaluatesIn.lookup rfl) (C.EvaluatesIn.lookup rfl) rfl)
        C.EvaluatesIn.constant rfl
    · exact C.EvaluatesIn.constant
    · change (some (.ok (.scalar (.int (((oddValue index : Int) * oddValue index - 3) / 2))), heap) :
        Option (Except ExceptionObject Mutable.Value × Mutable.Heap)) = _
      rw [← squareIndex_cast]
  have third : C.BlockContinuesIn program e2 [writeBody[0]'(by decide)] heap 19 e3 heap := by
    apply C.BlockContinuesIn.assign
    apply C.EvaluatesIn.binary (av := .scalar (.int (((bytes.length : Int) - start - 1) / p)))
      (bv := .scalar (.int 1)) (middle := heap) (afterRight := heap)
    · apply C.EvaluatesIn.binary (av := .scalar (.int ((bytes.length : Int) - start - 1)))
        (bv := .scalar (.int p)) (middle := heap) (afterRight := heap)
      · exact C.EvaluatesIn.binary
          (C.EvaluatesIn.binary
            (C.EvaluatesIn.lookup (by simpa [e2, e1, HeapCalls.setLocal] using getSize))
            (C.EvaluatesIn.lookup rfl) rfl) C.EvaluatesIn.constant rfl
      · exact C.EvaluatesIn.lookup (by simp [e2, e1, HeapCalls.setLocal])
      · simp [Mutable.binary, scalarBinary, Nat.ne_of_gt positive, positive]
    · exact C.EvaluatesIn.constant
    · rfl
  have fourth : C.BlockContinuesIn program e3 [writeBody[1]'(by decide)] heap
      (5 + count.toNat + 8 + 4 + (List.replicate count.toNat (0 : UInt8)).length) e3
      (heap.set address (.bytearray (mark bytes index))) := by
    apply C.BlockContinuesIn.setSlice (replacement := List.replicate count.toNat 0)
      (cr := 5 + count.toNat) (ct := 8) (afterRhs := heap) (afterTarget := heap)
      (before := bytes) (lowerValue := .scalar (.int start))
      (upperValue := .scalar .none) (strideValue := .scalar (.int p))
    · apply C.EvaluatesIn.binaryCost C.EvaluatesIn.constant (C.EvaluatesIn.lookup rfl)
        (by simp [Mutable.binary, scalarBinary])
      simp [C.resultBytes]
    · exact .cons (C.EvaluatesIn.lookup (by simpa [e3, e2, e1, HeapCalls.setLocal] using getSieve))
        (.cons (C.EvaluatesIn.lookup (by simp [e3, e2, HeapCalls.setLocal]))
          (.cons C.EvaluatesIn.constant (.cons (C.EvaluatesIn.lookup
            (by simp [e3, e2, e1, HeapCalls.setLocal])) .nil)))
    · exact found
    · exact mark_length bytes index
    · apply Mutable.setSlice_buffer heap address bytes _ _ _ _ _
        { start := some start, step := some p } found
      · change (if (p : Int) = 0 then some (.error ⟨"ValueError", []⟩, heap)
          else some (.ok { start := some (start : Int), stop := none, step := some (p : Int) }, heap) :
            Option (Except ExceptionObject FormalSW.Py.Builtin.Slice.Slice × Mutable.Heap)) = _
        rw [if_neg (by omega)]
      · exact mark_slice bytes index inside
  have checked : C.BlockContinuesIn program e2 [markBody[2]'(by decide)] heap
      (5 + (19 + (5 + count.toNat + 8 + 4 + (List.replicate count.toNat (0 : UInt8)).length)) + 2)
      (markEnv env index bytes.length) (heap.set address (.bytearray (mark bytes index))) := by
    apply C.BlockContinuesIn.branch (flag := true) (value := .scalar (.bool true)) (middle := heap)
    · apply C.EvaluatesIn.compare (C.EvaluatesIn.lookup rfl)
        (C.EvaluatesIn.lookup (by simpa [e2, e1, HeapCalls.setLocal] using getSize))
      simp [HeapCalls.compare, scalarCompare, evalCmpOp, start, inside]
    · rfl
    · exact third.append fourth
  simpa only [List.length_replicate, Nat.mul_add, Nat.mul_one, Nat.two_mul,
    Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using first.append (second.append checked)

end solvers.«10».complexity
