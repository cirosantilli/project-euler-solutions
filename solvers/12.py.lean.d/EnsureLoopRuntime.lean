import solvers.«12.py.lean.d».EnsureRuntime
import ProjectEulerStatements.Lib.PrimeCacheBounds

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

private theorem heap_set_same (heap : Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  obtain ⟨valid, value⟩ := List.getElem?_eq_some_iff.mp found
  rw [← value, List.set_getElem_self valid]

theorem ensure_loop (target bound candidate : Nat) (primeBound : Nat.Prime bound)
    (enough : target ≤ bound) (heap : Mutable.Heap) (address storage : Nat)
    (ps : List Nat) (last : Nat) (env : HeapCalls.Env)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps)))
    (lastEntry : ps.getLast? = some last) (ordered : ps.Pairwise (· < ·))
    (complete : ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps candidate)
    (lastBound : last ≤ bound)
    (large : 3 ≤ candidate) (odd : candidate % 2 = 1)
    (self : env "self" = some (.ref address))
    (getN : env "n" = some (.scalar (.int target)))
    (getCandidate : env "candidate" = some (.scalar (.int candidate))) :
    ∃ finalPrimes finalLast, finalPrimes.getLast? = some finalLast ∧ target ≤ finalLast ∧ finalLast ≤ bound ∧
      finalPrimes.Pairwise (· < ·) ∧
      ProjectEulerStatements.PrimeTrialDivision.CompleteBelow finalPrimes (finalLast + 1) ∧
      BlockReturnsIn .word program env [ensureLoop] heap
        ((41*(bound+1)+150)*(bound+2-candidate)+38) (.scalar .none)
        (heap.set storage (.list (primeValues finalPrimes))) := by
  have weakOrder := ordered.imp (fun h => Nat.le_of_lt h)
  have table : TableView heap address ps := ⟨storage, object, found⟩
  have tested : EvaluatesIn .word program env (.compare lastExpression [(.lt, .name "n")]) heap 35
      (.scalar (.bool (decide (last < target)))) heap := by
    apply EvaluatesIn.compare (work := 2) (table_last heap address ps last table lastEntry env self)
      (EvaluatesIn.lookup getN)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  by_cases short : last < target
  · let nextPrimes := if Nat.Prime candidate then ps ++ [candidate] else ps
    let nextLast := if Nat.Prime candidate then candidate else last
    let nextHeap := heap.set storage (.list (primeValues nextPrimes))
    let nextEnv := HeapCalls.setLocal env "candidate" (.scalar (.int (candidate + 2 : Nat)))
    have lastNext : nextPrimes.getLast? = some nextLast := by
      by_cases prime : Nat.Prime candidate <;> simp [nextPrimes, nextLast, prime, lastEntry]
    have distinct : storage ≠ address := by
      rintro rfl
      rw [object] at found
      cases found
    have nextObject : nextHeap[address]? =
        some (.instance origin "PrimeTable" [("primes", .ref storage)]) := by
      simpa [nextHeap, distinct] using object
    have nextFound : nextHeap[storage]? = some (.list (primeValues nextPrimes)) := by
      simp [nextHeap, (List.getElem?_eq_some_iff.mp found).1]
    have nextOrder := ProjectEulerStatements.PrimeTrialDivision.advance_ordered candidate ps ordered complete
    have nextComplete := ProjectEulerStatements.PrimeTrialDivision.advance_complete candidate ps large odd complete
    have bounded := ProjectEulerStatements.PrimeTrialDivision.candidate_le_prime ps last candidate target bound
      weakOrder lastEntry complete short enough primeBound
    have nextBound : nextLast ≤ bound := by
      by_cases prime : Nat.Prime candidate <;> simp [nextLast, prime] <;> omega
    obtain ⟨finalPrimes, finalLast, finalEntry, finalEnough, finalBound, finalOrder, finalComplete, rest⟩ :=
      ensure_loop target bound (candidate + 2) primeBound enough nextHeap address storage
        nextPrimes nextLast nextEnv nextObject nextFound lastNext nextOrder nextComplete nextBound
        (by omega) (by omega)
        (by simpa [nextEnv, HeapCalls.setLocal] using self)
        (by simpa [nextEnv, HeapCalls.setLocal] using getN) rfl
    refine ⟨finalPrimes, finalLast, finalEntry, finalEnough, finalBound, finalOrder, finalComplete, ?_⟩
    have lengthBound : ps.length ≤ bound+1 :=
      ProjectEulerStatements.PrimeCacheBounds.length_le_of_pairwise_lt_bounded ps bound ordered
        (fun p member => by have below := ((complete p).mp member).2; omega)
    have widthBound : PrimitiveWork.bitLength (candidate : Int) ≤ bound+1 := by
      have width := Nat.log2_le_self candidate
      simp only [PrimitiveWork.bitLength, Int.natAbs_natCast]
      omega
    have step := ensure_step candidate heap address storage ps env object found self getCandidate
      (by omega) weakOrder complete
    have continued := step.returns rest
    have paid : BlockReturnsIn .word program env [ensureLoop] heap
        (35 + ((40*ps.length+PrimitiveWork.bitLength (candidate : Int)+100) +
          ((41*(bound+1)+150)*(bound+2-(candidate+2))+38)) + 2)
        (.scalar .none) (heap.set storage (.list (primeValues finalPrimes))) := by
      apply BlockReturnsIn.while_loop (cc := 35) (flag := true) (by simpa [short] using tested) rfl
      simpa only [nextHeap, List.set_set] using continued
    apply paid.mono
    have measure : bound+2-candidate = (bound+2-(candidate+2))+2 := by omega
    rw [measure]
    simp only [Nat.mul_add, Nat.mul_one]
    omega
  · refine ⟨ps, last, lastEntry, by omega, lastBound, ordered,
      ProjectEulerStatements.PrimeTrialDivision.complete_at_last ps last candidate weakOrder lastEntry complete, ?_⟩
    rw [heap_set_same heap storage _ found]
    apply BlockReturnsIn.mono (small := 38) (larger := by omega)
    exact BlockReturnsIn.while_loop (cc := 35) (cb := 1) (flag := false)
      (by simpa [short] using tested) rfl BlockReturnsIn.fallthrough
termination_by bound + 2 - candidate
decreasing_by omega


end solvers.«12».runtimeCost
