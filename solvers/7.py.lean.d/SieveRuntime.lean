import solvers.«7.py.lean.d».MarkingRuntime
import Mathlib.Algebra.BigOperators.Intervals

namespace solvers.«7».sieveRuntime
open FormalSW.PyAst HeapCalls.Runtime ProjectEulerSolutions.P7
open markingRuntime
open scoped BigOperators
set_option maxRecDepth 4096
set_option maxHeartbeats 1000000
private abbrev program := Solvers.P7Ast.module

def iterationBlock : List Stmt :=
  match findFunctionIn program.body "solve" with
  | some (_, body) => match body.getLast? with
    | some (.whileDo _ attempt _) => match (attempt[5]? : Option Stmt) with
      | some (.forIn _ _ body _) => body
      | _ => []
    | _ => []
  | none => []

def candidateWork (size index : Nat) : Nat := 6 * markingWork.writes size index + emptyWork size index + 316
def loopWork (size first stop : Nat) : Nat := (∑ index ∈ Finset.Ico first stop, candidateWork size index) + 4

private theorem heap_set_same (heap : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  obtain ⟨valid, value⟩ := List.getElem?_eq_some_iff.mp found
  rw [← value, List.set_getElem_self valid]

theorem read (env : HeapCalls.Env) (heap : Mutable.Heap) (address index : Nat) (sieve : Array Bool)
    (valid : index < sieve.size)
    (getIndex : env "i" = some (.scalar (.int index)))
    (getSieve : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    EvaluatesIn .word program env (.subscript (.name "is_prime") (.name "i")) heap
      10 (.scalar (.int (if sieve[index] then 1 else 0))) heap := by
  apply EvaluatesIn.buffer_subscript (by intros a b c impossible; cases impossible)
    (EvaluatesIn.lookup getSieve) (EvaluatesIn.lookup getIndex) found
  have read := Mutable.getItem_buffer heap address index (sieveBytes sieve) found (by simpa [sieveBytes] using valid)
  cases flag : sieve[index] <;> simpa [sieveBytes, flag] using read

theorem iteration (index : Nat) (sieve : Array Bool) (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat)
    (marker : 1 ≤ index) (valid : index < sieve.size)
    (getIndex : env "i" = some (.scalar (.int index)))
    (getSize : env "size" = some (.scalar (.int sieve.size)))
    (getSieve : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    ∃ finalEnv, BlockContinuesIn .word program env iterationBlock heap
      (candidateWork sieve.size index - 4) finalEnv
      (heap.set address (.bytearray (sieveBytes (if sieve[index] then markMultiples (oddAt index) sieve else sieve)))) ∧
      Stable env finalEnv := by
  have evaluated := read env heap address index sieve valid getIndex getSieve found
  cases flag : sieve[index] with
  | false =>
      refine ⟨env, ?_, Stable.refl env⟩
      simp only [Bool.false_eq_true, if_false, heap_set_same heap address _ found]
      exact (BlockContinuesIn.branch (yes := markingWork.sourceBlock) (no := []) evaluated
        (flag := false) (by simp [Mutable.truth, scalarTruth, flag]) BlockContinuesIn.nil).mono (by unfold candidateWork; omega)
  | true =>
      obtain ⟨finalEnv, marked, stable⟩ := source_mark env heap address index sieve marker getIndex getSize getSieve found
      refine ⟨finalEnv, ?_, stable⟩
      simp only [if_true]
      exact (BlockContinuesIn.branch (yes := markingWork.sourceBlock) (no := []) evaluated
        (flag := true) (by simp [Mutable.truth, scalarTruth, flag]) marked).mono (by unfold candidateWork; omega)

/-- Runtime of the actual marking range, with its exact resulting sieve. -/
theorem loop (limit maxI next : Nat) (sieve : Array Bool)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (address : Nat)
    (positive : 1 ≤ next) (bounded : next ≤ maxI+1) (validMax : maxI < sieve.size)
    (getSize : env "size" = some (.scalar (.int sieve.size)))
    (getSieve : env "is_prime" = some (.ref address))
    (found : heap[address]? = some (.bytearray (sieveBytes sieve))) :
    ∃ finalEnv, BlockContinuesIn .word program env
      [.rangeStepLoop "i" next (maxI+1 : Nat) 1 iterationBlock []] heap
      (loopWork sieve.size next (maxI+1)) finalEnv
      (heap.set address (.bytearray (sieveBytes (sieveLoop limit maxI next sieve)))) ∧ Stable env finalEnv := by
  by_cases active : next ≤ maxI
  · have valid : next < sieve.size := by omega
    let indexed := HeapCalls.setLocal env "i" (.scalar (.int next))
    let updated := if sieve[next] then markMultiples (oddAt next) sieve else sieve
    have sizeUpdated : updated.size = sieve.size := by
      dsimp [updated]
      split <;> simp [markMultiples_size]
    have indexStable := Stable.index env next
    obtain ⟨middleEnv, marked, stable⟩ := iteration next sieve indexed heap address positive valid rfl
      ((indexStable "size" (by simp)).trans getSize) ((indexStable "is_prime" (by simp)).trans getSieve) found
    have addressValid := (List.getElem?_eq_some_iff.mp found).1
    obtain ⟨finalEnv, rest, finalStable⟩ := loop limit maxI (next+1) updated middleEnv
      (heap.set address (.bytearray (sieveBytes updated))) address (by omega) (by omega)
      (by rw [sizeUpdated]; exact validMax)
      (by rw [sizeUpdated]; exact (stable "size" (by simp)).trans ((indexStable "size" (by simp)).trans getSize))
      ((stable "is_prime" (by simp)).trans ((indexStable "is_prime" (by simp)).trans getSieve))
      (List.getElem?_set_self addressValid)
    refine ⟨finalEnv, ?_, indexStable.trans (stable.trans finalStable)⟩
    have combined := marked.append rest
    rw [sizeUpdated] at combined
    have entered := BlockContinuesIn.range_step (name := "i") (env := env) (next := (next : Int))
      (stop := ((maxI+1 : Nat) : Int)) (stride := 1)
      (by simp [Calls.rangeContinues]; omega)
      (by simpa only [Nat.cast_add, Nat.cast_one] using combined)
    have finalSieve : sieveLoop limit maxI next sieve = sieveLoop limit maxI (next+1) updated := by
      rw [sieveLoop.eq_def, if_neg (by omega)]
      simp only [getElem!_pos sieve next valid, updated]
      split <;> rfl
    rw [finalSieve]
    simp only [List.set_set] at entered
    apply entered.mono
    unfold loopWork
    rw [Finset.sum_eq_sum_Ico_succ_bot (show next < maxI+1 by omega)]
    unfold candidateWork
    omega
  · have equal : next = maxI+1 := by omega
    refine ⟨env, ?_, Stable.refl env⟩
    rw [sieveLoop.eq_def, if_pos (by omega), heap_set_same heap address _ found]
    have done := BlockContinuesIn.range_done (module := program) (env := env) (heap := heap)
      (name := "i") (next := (next : Int)) (stop := ((maxI+1 : Nat) : Int)) (stride := 1)
      (body := iterationBlock) (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
    simpa [loopWork, equal] using done
termination_by maxI+1-next
decreasing_by omega

theorem loopWork_bound (limit : Nat) :
    loopWork (sieveSize limit) 1 (Nat.sqrt limit / 2 + 1) ≤
      6 * sieveSize limit * ((sieveSize limit).log2+2) + 321*sieveSize limit + 4 := by
  have harmonic := markingWork.logarithmic_envelope limit (Finset.Ico 1 (Nat.sqrt limit/2+1)) (by
    intro i member
    have := Finset.mem_Ico.mp member
    exact Finset.mem_Icc.mpr (by omega))
  have empties : (∑ i ∈ Finset.Ico 1 (Nat.sqrt limit/2+1), emptyWork (sieveSize limit) i) ≤ 5*sieveSize limit := by
    calc
      _ ≤ ∑ i ∈ Finset.Ico 1 (Nat.sqrt limit/2+1),
          if i = Nat.sqrt limit/2 then 5*sieveSize limit else 0 := by
        apply Finset.sum_le_sum
        intro i member
        have := Finset.mem_Ico.mp member
        by_cases last : i = Nat.sqrt limit/2
        · simp only [last, if_true, emptyWork]
          split <;> omega
        · rw [if_neg last, emptyWork_before_last limit i (by omega)]
      _ ≤ _ := by simp; split <;> omega
  have root := Nat.sqrt_le_self limit
  have count : (Finset.Ico 1 (Nat.sqrt limit/2+1)).card ≤ sieveSize limit := by
    simp only [Nat.card_Ico]
    unfold sieveSize
    omega
  simp only [loopWork, candidateWork, Finset.sum_add_distrib, ← Finset.mul_sum,
    Finset.sum_const, nsmul_eq_mul]
  nlinarith only [harmonic, empties, count]

end solvers.«7».sieveRuntime
