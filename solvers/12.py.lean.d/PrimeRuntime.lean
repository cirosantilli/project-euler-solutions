import solvers.«12.py.lean.d».SetupRuntime
import lang.py.FormalSW.HeapRuntimeCostListLoops
import lang.py.FormalSW.HeapRuntimeCostLibrary

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

theorem prime_loop (n : Nat) (heap : Mutable.Heap) (storage : Nat) (ps : List Nat)
    (found : heap[storage]? = some (.list (primeValues ps)))
    (positive : ∀ p ∈ ps, 0 < p) (position : Nat) (env : HeapCalls.Env)
    (getX : env "x" = some (.scalar (.int n)))
    (getRoot : env "r" = some (.scalar (.int (Nat.sqrt n)))) :
    BlockReturnsIn .word program env
      [.sequenceStepLoop "p" (.ref storage) position primeIteration [], .return (.constant (.bool true))]
      heap (40*(ps.length-position)+7) (.scalar (.bool (ProjectEulerStatements.PrimeTrialDivision.trial n (ps.drop position)))) heap := by
  by_cases active : position < ps.length
  · let p := ps[position]
    let boundEnv := HeapCalls.setLocal env "p" (.scalar (.int p))
    have item : (primeValues ps)[position]? = some (.scalar (.int p)) := by
      simp only [primeValues, List.getElem?_map, List.getElem?_eq_getElem active, Option.map_some, p]
    have pPositive : 0 < p := positive p (List.getElem_mem active)
    have tested : EvaluatesIn .word program boundEnv
        (.compare (.name "p") [(.gt, .name "r")]) heap 7
        (.scalar (.bool (decide (Nat.sqrt n < p)))) heap := by
      apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup rfl)
        (EvaluatesIn.lookup (by simpa [boundEnv, HeapCalls.setLocal] using getRoot))
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
      · rfl
    rw [List.drop_eq_getElem_cons active]
    change BlockReturnsIn _ _ _ _ _ _
      (.scalar (.bool (ProjectEulerStatements.PrimeTrialDivision.trial n (p :: ps.drop (position + 1))))) _
    rw [ProjectEulerStatements.PrimeTrialDivision.trial]
    by_cases stop : Nat.sqrt n < p
    · rw [if_pos stop]
      apply BlockReturnsIn.mono (small := 17) (larger := by omega)
      apply BlockReturnsIn.list_step (budget := 12) found item
      apply BlockReturnsIn.branch (cc := 7) (cb := 3) (flag := true) (by simpa [stop] using tested) rfl
      exact BlockReturnsIn.break_loop rfl
        (BlockReturnsIn.return_value EvaluatesIn.constant)
    · rw [if_neg stop]
      have divisible : EvaluatesIn .word program boundEnv
          (.compare (.binOp (.name "x") .mod (.name "p")) [(.eq, .constant (.int 0))]) heap 12
          (.scalar (.bool (decide (n % p = 0)))) heap := by
        apply EvaluatesIn.compare (work := 2) (av := .scalar (.int (n % p))) (middle := heap)
          (EvaluatesIn.binary (op := .mod) (work := 1)
            (EvaluatesIn.lookup (by simpa [boundEnv, HeapCalls.setLocal] using getX))
            (EvaluatesIn.lookup rfl) (by simp [Mutable.binary, scalarBinary, pPositive, pPositive.ne']) rfl)
          EvaluatesIn.constant
        · simp [HeapCalls.compare, scalarCompare, evalCmpOp, ← Int.dvd_iff_emod_eq_zero, Int.natCast_dvd_natCast, Nat.dvd_iff_mod_eq_zero]
        · rfl
      by_cases divides : n % p = 0
      · rw [if_pos divides]
        apply BlockReturnsIn.mono (small := 30) (larger := by omega)
        apply BlockReturnsIn.list_step (budget := 25) found item
        apply BlockReturnsIn.branch (cc := 7) (cb := 16) (flag := false) (by simpa [stop] using tested) rfl
        exact BlockReturnsIn.branch (cc := 12) (cb := 2) (flag := true) (by simpa [divides] using divisible) rfl
          (BlockReturnsIn.return_value EvaluatesIn.constant)
      · rw [if_neg divides]
        have rest := prime_loop n heap storage ps found positive (position + 1) boundEnv
          (by simpa [boundEnv, HeapCalls.setLocal] using getX)
          (by simpa [boundEnv, HeapCalls.setLocal] using getRoot)
        apply BlockReturnsIn.mono (small := 29+(40*(ps.length-(position+1))+7)) (larger := by omega)
        apply BlockReturnsIn.mono (small := (7+(12+((40*(ps.length-(position+1))+7)+1)+2)+2)+5) (larger := by omega)
        apply BlockReturnsIn.list_step found item
        apply BlockReturnsIn.branch (cc := 7) (cb := 12+((40*(ps.length-(position+1))+7)+1)+2)
          (flag := false) (by simpa [stop] using tested) rfl
        apply BlockReturnsIn.branch (cc := 12) (cb := (40*(ps.length-(position+1))+7)+1)
          (flag := false) (by simpa [divides] using divisible) rfl
        exact BlockReturnsIn.loop_resume rest
  · rw [List.drop_eq_nil_of_le (by omega : ps.length ≤ position)]
    apply BlockReturnsIn.mono (small := 7) (larger := by omega)
    exact BlockReturnsIn.list_done found
      (List.getElem?_eq_none (by simpa [primeValues] using (show ps.length ≤ position by omega)))
      (BlockReturnsIn.return_value EvaluatesIn.constant)
termination_by ps.length - position
decreasing_by omega


/-- A field read copies only the prime-list reference. -/
theorem table_field (heap : Heap) (address storage : Nat) (env : HeapCalls.Env)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (self : env "self" = some (.ref address)) :
    EvaluatesIn .word program env (.attribute (.name "self") "primes") heap 17 (.ref storage) heap := by
  apply EvaluatesIn.field (budget := 1) (work := 14) (EvaluatesIn.lookup self)
  · have defined : program.findClass "PrimeTable" =
        some (origin, (program.findClassBody "PrimeTable").getD []) := rfl
    simp [getAttribute, instanceClass, object, defined]
  · simp [RecordWork.field, object, RecordWork.lookup]
    rfl

/-- The square-root library call retains its bit-length tariff even with fixed
arithmetic. Early breaks and returns stop the live list iterator. -/
theorem prime_method_cost (n : Nat) (heap : Heap) (address storage : Nat) (ps : List Nat)
    (object : heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]))
    (found : heap[storage]? = some (.list (primeValues ps)))
    (positive : ∀ p ∈ ps, 0 < p) :
    MethodReturnsIn .word program (.ref address) primeMethod.1 primeMethod.2 [.scalar (.int n)] heap
      (40*ps.length+PrimitiveWork.bitLength (n : Int)+37)
      (.scalar (.bool (ProjectEulerStatements.PrimeTrialDivision.trial n ps))) heap := by
  let env := functionEnv primeMethod.1 primeMethod.2 [.ref address, .scalar (.int n)] []
  have root : Math.isqrt n = Nat.sqrt n := Nat.eq_sqrt.mpr (Math.isqrt_spec n)
  have library : ReturnsIn .word program "isqrt" [.scalar (.int n)] heap
      (PrimitiveWork.bitLength (n : Int)+1) (.scalar (.int (Nat.sqrt n))) heap := by
    apply ReturnsIn.library (values := [.int n]) rfl rfl
    · change scalarResult (some (Math.isqrtCall [.int n])) heap = _
      simp [scalarResult, Math.isqrtCall, root, show ¬ (n : Int) < 0 by omega]
    · rfl
  have calculated : EvaluatesIn .word program env (.call (.name "isqrt") [.name "x"]) heap
      (PrimitiveWork.bitLength (n : Int)+4) (.scalar (.int (Nat.sqrt n))) heap := by
    exact (EvaluatesIn.call (ArgumentsIn.cons (EvaluatesIn.lookup rfl) ArgumentsIn.nil) library).mono (by omega)
  let withRoot := setLocal env "r" (.scalar (.int (Nat.sqrt n)))
  have field := table_field heap address storage withRoot object rfl
  have loop := prime_loop n heap storage ps found positive 0 withRoot rfl rfl
  have entered := BlockReturnsIn.for_list_field field found loop
  have body : BlockReturnsIn .word program env (executableBody primeMethod.2) heap
      (40*ps.length+PrimitiveWork.bitLength (n : Int)+32)
      (.scalar (.bool (ProjectEulerStatements.PrimeTrialDivision.trial n ps))) heap := by
    exact (BlockReturnsIn.assign calculated (by simpa only [List.drop_zero] using entered)).mono
      (by simp only [Nat.sub_zero]; omega)
  have run := MethodReturnsIn.of_body (receiver := .ref address) (values := [.scalar (.int n)]) rfl body
  intro anchors stack
  exact (run anchors stack).mono (by change 5+(40*ps.length+PrimitiveWork.bitLength (n : Int)+32) ≤ _; omega)

end solvers.«12».runtimeCost
