import solvers.«21.py.lean.d».SieveRuntime
import lang.py.FormalSW.HeapRuntimeCostListCopies
import lang.py.FormalSW.HeapRuntimeCostListRepetition
import ProjectEulerStatements.Lib.ReciprocalSum
import Mathlib.Algebra.BigOperators.Intervals

/-! The source sieve's outer loop composes its actual divisor passes. Summing
their integer reciprocal bounds gives a logarithmic envelope without assuming
anything about the contents of the divisor-sum table. -/
namespace solvers.«21».runtimeCost.Sieve
open FormalSW.PyAst HeapCalls.Runtime
open scoped BigOperators

/-- Select the outer `for` statement from the generated function body. -/
def outerLoop : List Stmt := ((executableBody definition.2).drop 1).take 1

def initialization : List Stmt := (executableBody definition.2).take 1

theorem outerBody_continuation (next : Stmt) : Calls.loopContinuation outerBody next = next := rfl

/-- A divisor pass costs 28 units beyond its inner iterations; entering its
outer range iteration costs four more. -/
def divisorWork (maximum i : Nat) : Nat := 20 * ((maximum + 1) / i + 1) + 32

def outerWork (maximum start : Nat) : Nat :=
  (∑ i ∈ Finset.Ico start (maximum / 2 + 1), divisorWork maximum i) + 4

theorem divisor_count_covers (maximum i : Nat) (positive : 0 < i) :
    maximum + 1 ≤ 2 * i + ((maximum + 1) / i + 1) * i := by
  have division : ((maximum + 1) / i) * i + (maximum + 1) % i = maximum + 1 := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod (maximum + 1) i
  have remainder := Nat.mod_lt (maximum + 1) positive
  simp only [Nat.add_mul, Nat.one_mul]
  omega

private theorem restore_same (heap : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  obtain ⟨bounded, equal⟩ := List.getElem?_eq_some_iff.mp found
  rw [← equal]
  exact List.set_getElem_self bounded

/-- All remaining positive divisors execute on the original range machine.
Only the table object and locals `i`/`j` can change. -/
theorem outer_loop (maximum start : Nat) (positive : 0 < start)
    (bounded : start ≤ maximum / 2 + 1)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (size : values.length = maximum + 1)
    (bound : env "n_max" = some (number maximum))
    (array : env "s" = some (.ref address))
    (found : heap[address]? = some (.list (encode values))) :
    ∃ (finalValues : List Int) (finalEnv : HeapCalls.Env),
      finalValues.length = values.length ∧
      (∀ name, name ≠ "i" → name ≠ "j" → finalEnv name = env name) ∧
      BlockContinuesIn .word program env
        [.rangeStepLoop "i" start (maximum / 2 + 1 : Nat) 1 outerBody []]
        heap (outerWork maximum start) finalEnv
        (heap.set address (.list (encode finalValues))) := by
  by_cases active : start < maximum / 2 + 1
  · let indexed := HeapCalls.setLocal env "i" (number start)
    have indexedBound : indexed "n_max" = some (number maximum) := by
      simpa [indexed, HeapCalls.setLocal] using bound
    have indexedArray : indexed "s" = some (.ref address) := by
      simpa [indexed, HeapCalls.setLocal] using array
    obtain ⟨middleValues, middleEnv, middleSize, passPreserved, pass⟩ :=
      divisor_pass ((maximum + 1) / start + 1) maximum start positive
        (divisor_count_covers maximum start positive) heap indexed address values size
        indexedBound indexedArray (by simp [indexed, HeapCalls.setLocal]) found
    let middleHeap := heap.set address (.list (encode middleValues))
    have middleFound : middleHeap[address]? = some (.list (encode middleValues)) :=
      List.getElem?_set_self (List.getElem?_eq_some_iff.mp found).1
    have middleBound : middleEnv "n_max" = some (number maximum) :=
      (passPreserved "n_max" (by decide)).trans indexedBound
    have middleArray : middleEnv "s" = some (.ref address) :=
      (passPreserved "s" (by decide)).trans indexedArray
    obtain ⟨finalValues, finalEnv, finalSize, restPreserved, rest⟩ :=
      outer_loop maximum (start + 1) (by omega) (by omega) middleHeap middleEnv address
        middleValues (middleSize.trans size) middleBound middleArray middleFound
    refine ⟨finalValues, finalEnv, finalSize.trans middleSize, ?_, ?_⟩
    · intro name notI notJ
      rw [restPreserved name notI notJ, passPreserved name notJ]
      simp [indexed, HeapCalls.setLocal, notI]
    · have remainder : BlockContinuesIn .word program middleEnv
          [.rangeStepLoop "i" ((start : Int) + 1) (maximum / 2 + 1 : Nat) 1 outerBody []]
          middleHeap (outerWork maximum (start + 1)) finalEnv
          (heap.set address (.list (encode finalValues))) := by
        simpa only [Nat.cast_add, Nat.cast_one, middleHeap, List.set_set] using rest
      have body := pass.append remainder
      have run := BlockContinuesIn.range_step (env := env) (name := "i")
        (next := (start : Int)) (stop := (maximum / 2 + 1 : Nat)) (stride := 1)
        (body := outerBody) (otherwise := [])
        (by simp [Calls.rangeContinues]; omega)
        (by simpa only [outerBody_continuation, indexed, number] using body)
      apply run.mono
      unfold outerWork
      rw [Finset.sum_eq_sum_Ico_succ_bot active]
      unfold divisorWork
      omega
  · have finished : start = maximum / 2 + 1 := by omega
    refine ⟨values, env, rfl, fun _ _ _ => rfl, ?_⟩
    rw [restore_same heap address _ found]
    simpa [outerWork, finished] using
      (BlockContinuesIn.range_done (module := program) (env := env) (heap := heap)
        (name := "i") (next := (start : Int)) (stop := (maximum / 2 + 1 : Nat)) (stride := 1)
        (body := outerBody) (otherwise := [])
        (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil)
termination_by maximum / 2 + 1 - start
decreasing_by omega

/-- Reuse the shared reciprocal-sum bound, including empty-pass overhead. -/
theorem outerWork_bound (maximum : Nat) :
    outerWork maximum 1 ≤
      20 * (maximum + 1) * ((maximum + 1).log2 + 2) + 32 * (maximum + 1) + 4 := by
  have within : maximum / 2 ≤ maximum + 1 := by omega
  have harmonic := ProjectEulerStatements.ReciprocalSum.with_overhead
    (maximum + 1) (maximum / 2) within
  have sumShift : (∑ i ∈ Finset.Ico 1 (maximum / 2 + 1), divisorWork maximum i) =
      ∑ k ∈ Finset.range (maximum / 2), divisorWork maximum (k + 1) := by
    simpa only [Nat.add_sub_cancel, Nat.add_sub_cancel_left, Nat.add_comm] using
      Finset.sum_Ico_eq_sum_range (divisorWork maximum) 1 (maximum / 2 + 1)
  unfold outerWork
  rw [sumShift]
  simp only [divisorWork, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const,
    Finset.card_range, nsmul_eq_mul]
  simp only [Finset.sum_add_distrib, Finset.sum_const, Finset.card_range, nsmul_eq_mul] at harmonic
  nlinarith only [harmonic, within]

/-- The actual outer `for` statement pays for `n_max // 2 + 1` and its
range entry in addition to all the divisor passes. -/
theorem source_outer_loop (maximum : Nat)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (address : Nat) (values : List Int)
    (size : values.length = maximum + 1)
    (bound : env "n_max" = some (number maximum))
    (array : env "s" = some (.ref address))
    (found : heap[address]? = some (.list (encode values))) :
    ∃ (finalValues : List Int) (finalEnv : HeapCalls.Env),
      finalValues.length = values.length ∧
      (∀ name, name ≠ "i" → name ≠ "j" → finalEnv name = env name) ∧
      BlockContinuesIn .word program env outerLoop heap
        (20 * (maximum + 1) * ((maximum + 1).log2 + 2) + 32 * (maximum + 1) + 23)
        finalEnv (heap.set address (.list (encode finalValues))) := by
  obtain ⟨finalValues, finalEnv, finalSize, preserved, loop⟩ :=
    outer_loop maximum 1 (by decide) (by omega) heap env address values size bound array found
  refine ⟨finalValues, finalEnv, finalSize, preserved, ?_⟩
  have divided : EvaluatesIn .word program env
      (.binOp (.name "n_max") .floorDiv (.constant (.int 2))) heap 6
      (number (maximum / 2 : Nat)) heap := by
    apply EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup bound) EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, number]
    · rfl
  have stop : EvaluatesIn .word program env
      (.binOp (.binOp (.name "n_max") .floorDiv (.constant (.int 2))) .add (.constant (.int 1)))
      heap 11 (number (maximum / 2 + 1 : Nat)) heap := by
    apply EvaluatesIn.binary (work := 1) divided EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, number]
    · rfl
  have run := BlockContinuesIn.for_range_pair (name := "i")
    (loop := outerBody) (otherwise := [])
    (EvaluatesIn.constant (value := .int 1)) stop loop
  change BlockContinuesIn .word program env
    [.forIn (.name "i") (.call (.name "range")
      [.constant (.int 1), .binOp (.binOp (.name "n_max") .floorDiv (.constant (.int 2)))
        .add (.constant (.int 1))]) outerBody []] heap _ finalEnv _
  apply run.mono
  have upper := outerWork_bound maximum
  omega

/-- Allocate the source seed list and repeated zero table, then bind `s`.
Both allocations and their retained dynamic-array credits are included. -/
theorem source_initialize (maximum : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (bound : env "n_max" = some (number maximum)) :
    BlockContinuesIn .word program env initialization heap (2 * (maximum + 1) + 20)
      (HeapCalls.setLocal env "s" (.ref (heap.length + 1)))
      (heap ++ [.list [number 0], .list (encode (List.replicate (maximum + 1) 0))]) := by
  let seedHeap := heap ++ [.list [number 0]]
  have seed : EvaluatesIn .word program env (.list [.constant (.int 0)]) heap 7
      (.ref heap.length) seedHeap := by
    exact EvaluatesIn.list (ArgumentsIn.cons EvaluatesIn.constant ArgumentsIn.nil)
  have count : EvaluatesIn .word program env
      (.binOp (.name "n_max") .add (.constant (.int 1))) seedHeap 6
      (number (maximum + 1 : Nat)) seedHeap := by
    apply EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup bound) EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, number]
    · rfl
  have found : seedHeap[heap.length]? = some (.list [number 0]) := by simp [seedHeap]
  have repeated := EvaluatesIn.list_repeat (items := [number 0]) seed count found
  have allocated : EvaluatesIn .word program env
      (.binOp (.list [.constant (.int 0)]) .mult
        (.binOp (.name "n_max") .add (.constant (.int 1)))) heap
      (2 * (maximum + 1) + 18) (.ref (heap.length + 1))
      (heap ++ [.list [number 0], .list (encode (List.replicate (maximum + 1) 0))]) := by
    convert repeated using 1 <;>
      simp [seedHeap, encode, number, PrimitiveWork.integer, List.append_assoc]; omega
  exact BlockContinuesIn.assign allocated

end solvers.«21».runtimeCost.Sieve
