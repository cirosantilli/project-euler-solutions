import solvers.«12.py.lean.d».EnsureLoopRuntime

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

theorem ensure_initialize (n last : Nat) (heap : Mutable.Heap) (address : Nat) (ps : List Nat)
    (table : TableView heap address ps) (lastEntry : ps.getLast? = some last) (short : last < n)
    (env : HeapCalls.Env) (self : env "self" = some (.ref address))
    (getN : env "n" = some (.scalar (.int n))) :
    BlockContinuesIn .word program env ((executableBody ensureMethod.2).take 3) heap 94
      (HeapCalls.setLocal env "candidate"
        (.scalar (.int (ProjectEulerStatements.PrimeTrialDivision.nextCandidate last)))) heap := by
  let body := executableBody ensureMethod.2
  let assigned := HeapCalls.setLocal env "candidate" (.scalar (.int (last + 1 : Nat)))
  have first : BlockContinuesIn .word program env [body[0]'(by decide)] heap 37 env heap := by
    apply BlockContinuesIn.branch (cc := 35) (cb := 0) (flag := false) (value := .bool false) (middle := heap)
    · apply EvaluatesIn.compare (work := 2) (table_last heap address ps last table lastEntry env self)
        (EvaluatesIn.lookup getN)
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp, show ¬ n ≤ last by omega]
      · rfl
    · rfl
    · exact BlockContinuesIn.nil
  have second : BlockContinuesIn .word program env [body[1]'(by decide)] heap 36 assigned heap := by
    apply BlockContinuesIn.assign
    apply EvaluatesIn.binary (work := 1) (table_last heap address ps last table lastEntry env self)
      EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary]
    · rfl
  have evenTest : EvaluatesIn .word program assigned
      (.compare (.binOp (.name "candidate") .mod (.constant (.int 2))) [(.eq, .constant (.int 0))]) heap 12
      (.scalar (.bool (decide ((last + 1) % 2 = 0)))) heap := by
    apply EvaluatesIn.compare (work := 2) (av := .scalar (.int ((last + 1) % 2 : Nat))) (middle := heap)
      (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup rfl) EvaluatesIn.constant
        (by simp [Mutable.binary, scalarBinary]) rfl) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      omega
    · rfl
  have third : BlockContinuesIn .word program assigned [body[2]'(by decide)] heap 21
      (HeapCalls.setLocal env "candidate"
        (.scalar (.int (ProjectEulerStatements.PrimeTrialDivision.nextCandidate last)))) heap := by
    by_cases even : (last + 1) % 2 = 0
    · apply BlockContinuesIn.branch (cc := 12) (cb := 7) (flag := true) (by simpa [even] using evenTest) rfl
      have advanced : BlockContinuesIn .word program assigned
          [.augAssign (.name "candidate") .add (.constant (.int 1))] heap 7
          (HeapCalls.setLocal assigned "candidate" (.scalar (.int (last + 2 : Nat)))) heap := by
        apply BlockContinuesIn.augAssign (work := 1) (left := .int (last + 1 : Nat)) rfl
          EvaluatesIn.constant
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, Int.add_assoc]
        · rfl
      have overwritten : HeapCalls.setLocal assigned "candidate" (.scalar (.int (last + 2 : Nat))) =
          HeapCalls.setLocal env "candidate" (.scalar (.int (last + 2 : Nat))) := by
        funext query
        by_cases same : query = "candidate" <;> simp [assigned, HeapCalls.setLocal, same]
      rw [overwritten] at advanced
      simpa only [ProjectEulerStatements.PrimeTrialDivision.nextCandidate, if_pos even] using advanced
    · apply BlockContinuesIn.mono (small := 14) (larger := by omega)
      apply BlockContinuesIn.branch (cc := 12) (cb := 0) (flag := false) (by simpa [even] using evenTest) rfl
      simpa only [ProjectEulerStatements.PrimeTrialDivision.nextCandidate, if_neg even] using
        (BlockContinuesIn.nil (module := program) (env := assigned) (heap := heap))
  exact first.append (second.append third)


/-- An already sufficient cache pays only for its last-entry comparison. -/
theorem ensure_cached (n last : Nat) (heap : Heap) (address : Nat) (ps : List Nat)
    (table : TableView heap address ps) (lastEntry : ps.getLast? = some last) (enough : n ≤ last) :
    MethodReturnsIn .word program (.ref address) ensureMethod.1 ensureMethod.2
      [.scalar (.int n)] heap 44 (.scalar .none) heap := by
  apply MethodReturnsIn.of_body (budget := 39) rfl
  apply BlockReturnsIn.branch (cc := 35) (cb := 2) (flag := true) (value := .bool true) (middle := heap)
  · apply EvaluatesIn.compare (work := 2) (table_last heap address ps last table lastEntry _ rfl)
      (EvaluatesIn.lookup rfl)
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, enough]
    · rfl
  · rfl
  · exact BlockReturnsIn.return_value EvaluatesIn.constant

private theorem heap_set_same (heap : Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : heap.set address object = heap := by
  obtain ⟨valid, value⟩ := List.getElem?_eq_some_iff.mp found
  rw [← value, List.set_getElem_self valid]

/-- The same prime bound applies to every extension during the outer search. -/
theorem ensure_method_cost (n last bound : Nat) (primeBound : Nat.Prime bound)
    (enough : n ≤ bound) (lastBound : last ≤ bound)
    (heap : Heap) (address storage : Nat) (ps : List Nat)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps)))
    (lastEntry : ps.getLast? = some last) (ordered : ps.Pairwise (· < ·))
    (complete : ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps (last+1)) :
    ∃ finalPrimes finalLast, finalPrimes.getLast? = some finalLast ∧ n ≤ finalLast ∧ finalLast ≤ bound ∧
      finalPrimes.Pairwise (· < ·) ∧
      ProjectEulerStatements.PrimeTrialDivision.CompleteBelow finalPrimes (finalLast+1) ∧
      MethodReturnsIn .word program (.ref address) ensureMethod.1 ensureMethod.2 [.scalar (.int n)] heap
        ((41*(bound+1)+150)*(bound+2)+140) (.scalar .none)
        (heap.set storage (.list (primeValues finalPrimes))) := by
  have table : TableView heap address ps := ⟨storage, object, found⟩
  by_cases enoughNow : n ≤ last
  · refine ⟨ps, last, lastEntry, enoughNow, lastBound, ordered, complete, ?_⟩
    rw [heap_set_same heap storage _ found]
    intro anchors stack
    exact (ensure_cached n last heap address ps table lastEntry enoughNow anchors stack).mono (by omega)
  · have lastPrime : Nat.Prime last := ((complete last).mp (List.mem_of_mem_getLast? lastEntry)).1
    let candidate := ProjectEulerStatements.PrimeTrialDivision.nextCandidate last
    have bounds := ProjectEulerStatements.PrimeTrialDivision.nextCandidate_spec last lastPrime.two_le
    have readyComplete := ProjectEulerStatements.PrimeTrialDivision.nextCandidate_complete last ps lastPrime complete
    let env := functionEnv ensureMethod.1 ensureMethod.2 [.ref address, .scalar (.int n)] []
    let ready := setLocal env "candidate" (.scalar (.int candidate))
    obtain ⟨finalPrimes, finalLast, finalEntry, finalEnough, finalBound, finalOrder, finalComplete, loop⟩ :=
      ensure_loop n bound candidate primeBound enough heap address storage ps last ready object found
        lastEntry ordered readyComplete lastBound bounds.2.1 bounds.2.2 rfl rfl rfl
    refine ⟨finalPrimes, finalLast, finalEntry, finalEnough, finalBound, finalOrder, finalComplete, ?_⟩
    have init := ensure_initialize n last heap address ps table lastEntry (by omega) env rfl rfl
    have body : BlockReturnsIn .word program env (executableBody ensureMethod.2) heap
        (94+((41*(bound+1)+150)*(bound+2-candidate)+38)) (.scalar .none)
        (heap.set storage (.list (primeValues finalPrimes))) := init.returns loop
    have run := MethodReturnsIn.of_body (receiver := .ref address) (values := [.scalar (.int n)]) rfl body
    intro anchors stack
    apply (run anchors stack).mono
    have bounded := Nat.mul_le_mul_left (41*(bound+1)+150) (Nat.sub_le (bound+2) candidate)
    change 5+(94+((41*(bound+1)+150)*(bound+2-candidate)+38)) ≤ _
    omega

end solvers.«12».runtimeCost
