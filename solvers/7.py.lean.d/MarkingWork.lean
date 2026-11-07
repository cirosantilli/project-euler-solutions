import solvers.«7.py.lean.d».Source
import lang.py.FormalSW.BufferCostSlice
import ProjectEulerStatements.Lib.ReciprocalSum

/-! Source-linked slice marking work for one Euler 7 sieve attempt. These are
write-count envelopes, not full-trace bounds or bounds on the retry sequence. -/
namespace solvers.«7».markingWork
open FormalSW.PyAst ProjectEulerSolutions.P7
open scoped BigOperators

/-- Extract the marking branch inside the actual source's retry body. -/
def sourceBlock : List Stmt :=
  match findFunctionIn Solvers.P7Ast.module.body "solve" with
  | some (_, body) => match body.getLast? with
    | some (.whileDo _ attempt _) => match (attempt[5]? : Option Stmt) with
      | some (.forIn _ _ [.ifThen _ marking _] _) => marking
      | _ => []
    | _ => []
  | none => []

/-- The positive-stride slice positions used by `markMultiples_bytes`. -/
def positions (size candidate : Nat) : List Nat :=
  List.range' (oddAt candidate * oddAt candidate / 2)
    ((size - oddAt candidate * oddAt candidate / 2 + oddAt candidate - 1) / oddAt candidate)
    (oddAt candidate)

def writes (size candidate : Nat) : Nat := (positions size candidate).length

theorem source_count (size candidate : Nat) :
    (((size : Int) - 1 - (oddAt candidate * oddAt candidate / 2 : Nat)) / oddAt candidate + 1).toNat =
      writes size candidate := by
  unfold writes positions
  rw [List.length_range']
  exact FormalSW.Py.Builtin.Slice.positive_end_count _ _ _ (by unfold oddAt; omega)

/-- This is precisely the footprint used by the existing marking correctness proof. -/
theorem positions_update (sieve : Array Bool) (candidate : Nat) :
    (positions sieve.size candidate).foldl (fun bytes index => bytes.set index 0) (sieveBytes sieve) =
      sieveBytes (markMultiples (oddAt candidate) sieve) :=
  markMultiples_bytes candidate sieve

theorem writes_le (size candidate : Nat) : writes size candidate ≤ size / oddAt candidate + 1 := by
  unfold writes positions
  rw [List.length_range']
  have bound : size - oddAt candidate * oddAt candidate / 2 + oddAt candidate - 1 ≤ size + oddAt candidate := by omega
  calc
    _ ≤ (size + oddAt candidate) / oddAt candidate := Nat.div_le_div_right bound
    _ = _ := Nat.add_div_right _ (by unfold oddAt; omega)

/-- Pair the actual source-block execution with its repeated-byte length and
write-count envelope. Repeated writes to a zero byte are still charged. -/
theorem source_mark (env : HeapCalls.Env) (heap : Mutable.Heap) (address candidate : Nat) (sieve : Array Bool)
    (marker : 1 ≤ candidate)
    (getIndex : env "i" = some (.scalar (.int candidate)))
    (getSize : env "size" = some (.scalar (.int sieve.size)))
    (getSieve : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    ∃ finalEnv,
      HeapCalls.BlockContinues Solvers.P7Ast.module env sourceBlock heap finalEnv
        (heap.set address (.bytearray (sieveBytes (markMultiples (oddAt candidate) sieve)))) ∧
      finalEnv "count" = some (.scalar (.int (((sieve.size : Int) - 1 -
        (oddAt candidate * oddAt candidate / 2 : Nat)) / oddAt candidate + 1))) ∧
      (((sieve.size : Int) - 1 - (oddAt candidate * oddAt candidate / 2 : Nat)) / oddAt candidate + 1).toNat =
        writes sieve.size candidate ∧
      writes sieve.size candidate ≤ sieve.size / oddAt candidate + 1 := by
  refine ⟨_, solvers.«7».sieve.mark env heap address candidate sieve marker getIndex getSize getSieve found,
    rfl, source_count _ _, writes_le _ _⟩

/-- One attempt considers indices 1 through `isqrt(limit)//2`, inclusive. -/
theorem source_loop_envelope (limit : Nat) (selected : Finset Nat)
    (within : selected ⊆ Finset.Icc 1 (Nat.sqrt limit / 2)) :
    (∑ candidate ∈ selected, writes (sieveSize limit) candidate) ≤
      ∑ candidate ∈ Finset.Icc 1 (Nat.sqrt limit / 2), (sieveSize limit / oddAt candidate + 1) := by
  calc
    _ ≤ ∑ candidate ∈ selected, (sieveSize limit / oddAt candidate + 1) :=
      Finset.sum_le_sum (fun candidate _ => writes_le _ _)
    _ ≤ _ := Finset.sum_le_sum_of_subset within

/-- The sum over all candidates gives a logarithmic envelope for one attempt. -/
theorem logarithmic_envelope (limit : Nat) (selected : Finset Nat)
    (within : selected ⊆ Finset.Icc 1 (Nat.sqrt limit / 2)) :
    (∑ candidate ∈ selected, writes (sieveSize limit) candidate) ≤
      sieveSize limit * ((sieveSize limit).log2 + 2) := by
  have rangeWithin : selected ⊆ Finset.range (Nat.sqrt limit / 2 + 1) := by
    intro candidate member
    have bounds := Finset.mem_Icc.mp (within member)
    exact Finset.mem_range.mpr (by omega)
  calc
    _ ≤ ∑ candidate ∈ selected, (sieveSize limit / (candidate + 1) + 1) := by
      apply Finset.sum_le_sum
      intro candidate _
      have upper := writes_le (sieveSize limit) candidate
      have divisor := Nat.div_le_div_left
        (show candidate + 1 ≤ oddAt candidate by unfold oddAt; omega)
        (by omega : 0 < candidate + 1) (a := sieveSize limit)
      omega
    _ ≤ ∑ candidate ∈ Finset.range (Nat.sqrt limit / 2 + 1),
        (sieveSize limit / (candidate + 1) + 1) := Finset.sum_le_sum_of_subset rangeWithin
    _ ≤ _ := ProjectEulerStatements.ReciprocalSum.with_overhead _ _ (by
      have root := Nat.sqrt_le_self limit
      unfold sieveSize
      omega)

/-- Exact BufferCost charge for one successful slice-write transition. This
counts every selected byte, separately from RHS construction and loop overhead. -/
theorem slice_charge (anchors : Nat) (stack : List HeapCalls.Frame) (heap : Mutable.Heap)
    (address candidate : Nat) (sieve : Array Bool) (marker : 1 ≤ candidate)
    (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    HeapCalls.BufferCost.charge Solvers.P7Ast.module anchors
      ⟨.operation (.setSlice (.scalar (.bytes (List.replicate (writes sieve.size candidate) 0))))
        [.ref address, .scalar (.int (oddAt candidate * oddAt candidate / 2 : Nat)), .scalar .none,
         .scalar (.int (oddAt candidate))], stack, heap⟩ = 1 + writes sieve.size candidate := by
  have write := markSlice_buffer heap address candidate sieve marker found
  dsimp only at write
  rw [source_count] at write
  have length : (sieveBytes (markMultiples (oddAt candidate) sieve)).length = (sieveBytes sieve).length := by
    simp [sieveBytes, markMultiples_size]
  simpa only [List.length_replicate] using
    HeapCalls.BufferCost.slice_charge Solvers.P7Ast.module anchors stack heap address (sieveBytes sieve)
      (sieveBytes (markMultiples (oddAt candidate) sieve)) (List.replicate (writes sieve.size candidate) 0)
      _ _ _ found length write

end solvers.«7».markingWork
