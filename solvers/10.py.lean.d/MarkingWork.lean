import solvers.«10.py.lean.d».Source
import lang.py.FormalSW.BufferCostSlice
import ProjectEulerStatements.Lib.ReciprocalSum

/-! A source-linked marking-work envelope, not a bound on the full interpreter
trace. Work counts slice positions written, including writes of an already-zero
byte. Initialization, expression evaluation, traversal and summation are separate. -/
namespace solvers.«10».markingWork
open FormalSW.PyAst
open scoped BigOperators

/-- Extract the marking branch from the actual source AST. -/
def sourceBlock : List Stmt :=
  match findFunctionIn Solvers.P10Ast.module.body "solve" with
  | some (_, body) => match ((executableBody body)[6]? : Option Stmt) with
    | some (.forIn _ _ [.ifThen _ marking _] _) => marking
    | _ => []
  | none => []

/-- Number of positions in this marking event, not distinct bytes cleared. -/
def writes (size candidate : Nat) : Nat := (markedIndices size candidate).length

/-- The source's repeated zero-byte RHS has exactly one byte per selected position. -/
theorem source_count (size candidate : Nat) :
    (((size : Int) - squareIndex candidate - 1) / oddValue candidate + 1).toNat =
      writes size candidate := by
  unfold writes markedIndices
  rw [List.length_range']
  rw [show (size : Int) - squareIndex candidate - 1 = (size : Int) - 1 - squareIndex candidate by omega]
  exact FormalSW.Py.Builtin.Slice.positive_end_count _ _ _ (by unfold oddValue; omega)

theorem writes_le (size candidate : Nat) : writes size candidate ≤ size / oddValue candidate + 1 := by
  unfold writes markedIndices
  rw [List.length_range']
  have bound : size - squareIndex candidate + oddValue candidate - 1 ≤ size + oddValue candidate := by omega
  calc
    _ ≤ (size + oddValue candidate) / oddValue candidate := Nat.div_le_div_right bound
    _ = _ := Nat.add_div_right _ (by unfold oddValue; omega)

/-- The existing source execution proof supplies the exact state update whose
slice footprint is bounded here. This does not assert a weighted full-trace cost. -/
theorem source_mark (env : HeapCalls.Env) (heap : Mutable.Heap) (address candidate : Nat) (bytes : List UInt8)
    (inside : squareIndex candidate < bytes.length)
    (getIndex : env "i" = some (.scalar (.int candidate)))
    (getSize : env "size" = some (.scalar (.int bytes.length)))
    (getSieve : env "sieve" = some (.ref address))
    (found : heap[address]? = some (.bytearray bytes)) :
    ∃ finalEnv,
      HeapCalls.BlockContinues Solvers.P10Ast.module env sourceBlock heap finalEnv
        (heap.set address (.bytearray (mark bytes candidate))) ∧
      finalEnv "count" = some (.scalar (.int (((bytes.length : Int) - squareIndex candidate - 1) / oddValue candidate + 1))) ∧
      (((bytes.length : Int) - squareIndex candidate - 1) / oddValue candidate + 1).toNat = writes bytes.length candidate ∧
      writes bytes.length candidate ≤ bytes.length / oddValue candidate + 1 := by
  refine ⟨_, solvers.«10».mark.result env heap address candidate bytes inside getIndex getSize getSieve found,
    rfl, source_count _ _, writes_le _ _⟩

/-- Count every possible marking candidate. Any skipped composite only lowers
this envelope; the selection need not assume a prime-density estimate. -/
theorem envelope (size stop : Nat) (selected : Finset Nat) (within : selected ⊆ Finset.range stop) :
    (∑ candidate ∈ selected, writes size candidate) ≤
      ∑ candidate ∈ Finset.range stop, (size / oddValue candidate + 1) := by
  calc
    _ ≤ ∑ candidate ∈ selected, (size / oddValue candidate + 1) :=
      Finset.sum_le_sum (fun candidate _ => writes_le size candidate)
    _ ≤ _ := Finset.sum_le_sum_of_subset within

/-- The source loop's exact candidate range specializes the envelope. -/
theorem source_loop_envelope (limit : Nat) (selected : Finset Nat)
    (within : selected ⊆ Finset.range ((Nat.sqrt (limit - 1) - 1) / 2)) :
    (∑ candidate ∈ selected, writes (sieveSize limit) candidate) ≤
      ∑ candidate ∈ Finset.range ((Nat.sqrt (limit - 1) - 1) / 2),
        (sieveSize limit / oddValue candidate + 1) := envelope _ _ _ within

/-- A convenient harmonic envelope, without claiming a prime-reciprocal bound. -/
theorem harmonic_envelope (size stop : Nat) (selected : Finset Nat) (within : selected ⊆ Finset.range stop) :
    (∑ candidate ∈ selected, writes size candidate) ≤
      ∑ candidate ∈ Finset.range stop, (size / (candidate + 1) + 1) := by
  apply le_trans (envelope size stop selected within)
  apply Finset.sum_le_sum
  intro candidate _
  have := Nat.div_le_div_left (show candidate + 1 ≤ oddValue candidate by unfold oddValue; omega)
    (by omega : 0 < candidate + 1) (a := size)
  omega

/-- A closed envelope for all writes in the actual candidate loop. -/
theorem logarithmic_envelope (limit : Nat) (selected : Finset Nat)
    (within : selected ⊆ Finset.range ((Nat.sqrt (limit - 1) - 1) / 2)) :
    (∑ candidate ∈ selected, writes (sieveSize limit) candidate) ≤
      sieveSize limit * ((sieveSize limit).log2 + 2) := by
  apply le_trans (harmonic_envelope _ _ selected within)
  apply ProjectEulerStatements.ReciprocalSum.with_overhead
  have root := Nat.sqrt_le_self (limit - 1)
  unfold sieveSize
  omega

/-- Exact BufferCost charge at the successful source slice primitive. The RHS
construction has its own, separate charge; this is only the write transition. -/
theorem slice_charge (anchors : Nat) (stack : List HeapCalls.Frame) (heap : Mutable.Heap)
    (address candidate : Nat) (bytes : List UInt8) (inside : squareIndex candidate < bytes.length)
    (found : heap[address]? = some (.bytearray bytes)) :
    HeapCalls.BufferCost.charge Solvers.P10Ast.module anchors
      ⟨.operation (.setSlice (.scalar (.bytes (List.replicate (writes bytes.length candidate) 0))))
        [.ref address, .scalar (.int (squareIndex candidate)), .scalar .none, .scalar (.int (oddValue candidate))],
        stack, heap⟩ = 1 + writes bytes.length candidate := by
  have write : Mutable.setSlice (.ref address) (.scalar (.int (squareIndex candidate)))
      (.scalar .none) (.scalar (.int (oddValue candidate)))
      (.scalar (.bytes (List.replicate (writes bytes.length candidate) 0))) heap =
      some (.ok (.scalar .none), heap.set address (.bytearray (mark bytes candidate))) := by
    apply Mutable.setSlice_buffer heap address bytes _ _ _ _ _
      { start := some (squareIndex candidate), step := some (oddValue candidate) } found
    · change (if (oddValue candidate : Int) = 0 then some (.error ⟨"ValueError", []⟩, heap)
        else some (.ok { start := some (squareIndex candidate : Int), stop := none, step := some (oddValue candidate : Int) }, heap) :
          Option (Except ExceptionObject FormalSW.Py.Builtin.Slice.Slice × Mutable.Heap)) = _
      rw [if_neg (by unfold oddValue; omega)]
    · simpa only [source_count] using mark_slice bytes candidate inside
  simpa only [List.length_replicate] using
    HeapCalls.BufferCost.slice_charge Solvers.P10Ast.module anchors stack heap address bytes (mark bytes candidate)
      (List.replicate (writes bytes.length candidate) 0) _ _ _ found (mark_length bytes candidate) write

end solvers.«10».markingWork
