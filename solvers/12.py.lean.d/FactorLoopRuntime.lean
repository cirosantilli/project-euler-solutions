import solvers.«12.py.lean.d».StripRuntime
import solvers.«12.py.lean.d».FactorTailRuntime
import lang.py.FormalSW.HeapRuntimeCostListLoops

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
open ProjectEulerStatements.FactorDivisors

private def divisibleBody : List Stmt :=
  match (factorIteration[1]? : Option Stmt) with
  | some (.ifThen _ body _) => body | _ => []

/-- Scanning pays once per cached prime; all successful divisions share the
initial residue's decreasing budget. -/
theorem factor_loop (x total : Nat) (positive : 0 < x) (heap : Heap)
    (storage : Nat) (ps : List Nat) (found : heap[storage]? = some (.list (primeValues ps)))
    (primes : ∀ p ∈ ps, Nat.Prime p) (position : Nat) (env : HeapCalls.Env)
    (getX : env "x" = some (.scalar (.int x)))
    (getTotal : env "total" = some (.scalar (.int total))) :
    BlockReturnsIn .word program env
      (.sequenceStepLoop "p" (.ref storage) position factorIteration [] :: factorTail) heap
      (80*(ps.length-position)+40*x+40)
      (.scalar (.int (total * count x (ps.drop position) : Nat))) heap := by
  by_cases active : position < ps.length
  · let p := ps[position]
    let boundEnv := setLocal env "p" (.scalar (.int p))
    have prime : Nat.Prime p := primes p (List.getElem_mem active)
    have item : (primeValues ps)[position]? = some (.scalar (.int p)) := by
      simp only [primeValues, List.getElem?_map, List.getElem?_eq_getElem active, Option.map_some, p]
    have stopTest : EvaluatesIn .word program boundEnv
        (.compare (.binOp (.name "p") .mult (.name "p")) [(.gt, .name "x")]) heap 12
        (.scalar (.bool (decide (x < p*p)))) heap := by
      apply EvaluatesIn.compare (work := 2) (av := .scalar (.int (p*p : Nat)))
        (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup rfl) (EvaluatesIn.lookup rfl)
          (by simp [Mutable.binary, scalarBinary]) rfl)
        (EvaluatesIn.lookup (by simpa [boundEnv, setLocal] using getX))
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        norm_cast
      · rfl
    rw [List.drop_eq_getElem_cons active]
    change BlockReturnsIn _ _ _ _ _ _ (.scalar (.int (total * count x (p :: ps.drop (position+1)) : Nat))) _
    rw [count]
    by_cases stop : x < p*p
    · rw [if_pos stop]
      have finish := factor_finish x total heap boundEnv
        (by simpa [boundEnv, setLocal] using getX)
        (by simpa [boundEnv, setLocal] using getTotal)
      have branch : BlockReturnsIn .word program boundEnv
          (factorIteration ++ [.loopResume (.sequenceStepLoop "p" (.ref storage) (position+1) factorIteration [])] ++ factorTail)
          heap 33 (.scalar (.int (total * ProjectEulerStatements.FactorDivisors.finish x : Nat))) heap := by
        apply BlockReturnsIn.branch (cc := 12) (cb := 19) (flag := true)
          (by simpa [stop] using stopTest) rfl
        exact BlockReturnsIn.break_loop rfl finish
      exact (BlockReturnsIn.list_step (body := factorIteration) (otherwise := []) (following := factorTail) found item branch).mono (by omega)
    · rw [if_neg stop]
      have dividesTest : EvaluatesIn .word program boundEnv
          (.compare (.binOp (.name "x") .mod (.name "p")) [(.eq, .constant (.int 0))]) heap 12
          (.scalar (.bool (decide (p ∣ x)))) heap := by
        apply EvaluatesIn.compare (work := 2) (av := .scalar (.int (x%p)))
          (EvaluatesIn.binary (work := 1)
            (EvaluatesIn.lookup (by simpa [boundEnv, setLocal] using getX)) (EvaluatesIn.lookup rfl)
            (by simp [Mutable.binary, scalarBinary, prime.pos, prime.ne_zero]) rfl) EvaluatesIn.constant
        · simp [HeapCalls.compare, scalarCompare, evalCmpOp, ← Int.dvd_iff_emod_eq_zero, Int.natCast_dvd_natCast]
        · rfl
      let continuation := [.loopResume (.sequenceStepLoop "p" (.ref storage) (position+1) factorIteration [])] ++ factorTail
      by_cases divides : p ∣ x
      · let initialized := setLocal boundEnv "exp" (.scalar (.int 0))
        obtain ⟨stripped, stripping, resultX, resultExponent, saved⟩ := strip_cost x p 0 positive prime heap initialized
          (by simpa [initialized, boundEnv, setLocal] using getX)
          (by simp [initialized, boundEnv, setLocal]) rfl
        have totalSaved : stripped "total" = some (.scalar (.int total)) :=
          (saved "total" (by decide) (by decide)).trans
            (by simpa [initialized, boundEnv, setLocal] using getTotal)
        have rest := factor_loop (strip x p).1 (total*((strip x p).2+1)) (strip_spec x p positive prime).1
          heap storage ps found primes (position+1)
          (setLocal stripped "total" (.scalar (.int (total*((strip x p).2+1) : Nat))))
          (by simpa [setLocal] using resultX) rfl
        have exponent : EvaluatesIn .word program stripped (.binOp (.name "exp") .add (.constant (.int 1))) heap 6
            (.scalar (.int ((strip x p).2+1 : Nat))) heap := by
          apply EvaluatesIn.binary (work := 1)
            (EvaluatesIn.lookup (by simpa only [Nat.zero_add] using resultExponent)) EvaluatesIn.constant
          · simp [Mutable.binary, scalarBinary]
          · rfl
        have advance := BlockReturnsIn.augAssign (op := .mult) (work := 1) (value := .scalar (.int (total*((strip x p).2+1) : Nat))) totalSaved exponent
          (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary]) rfl (BlockReturnsIn.loop_resume rest)
        have prepared := BlockReturnsIn.assign (name := "exp") EvaluatesIn.constant (stripping.returns advance)
        have divisible := BlockReturnsIn.branch (cc := 12) (yes := divisibleBody) (no := []) (following := continuation) (flag := true)
          (by simpa [divides] using dividesTest) rfl prepared
        have continued := BlockReturnsIn.branch (cc := 12) (yes := [.breakLoop]) (no := []) (following := factorIteration.drop 1 ++ continuation) (flag := false)
          (by simpa [stop] using stopTest) rfl divisible
        have run := BlockReturnsIn.list_step (body := factorIteration) (otherwise := []) (following := factorTail) found item continued
        have residue := strip_le x p positive prime
        simpa only [Nat.mul_assoc] using run.mono (show _ ≤ 80*(ps.length-position)+40*x+40 by omega)
      · have unchanged : strip x p = (x,0) := by
          rw [strip, dif_neg (by tauto)]
        simp only [unchanged, Nat.zero_add, Nat.one_mul]
        have rest := factor_loop x total positive heap storage ps found primes (position+1) boundEnv
          (by simpa [boundEnv, setLocal] using getX)
          (by simpa [boundEnv, setLocal] using getTotal)
        have branch := BlockReturnsIn.branch (cc := 12) (yes := divisibleBody) (no := []) (following := continuation) (flag := false)
          (by simpa [divides] using dividesTest) rfl (BlockReturnsIn.loop_resume rest)
        have continued := BlockReturnsIn.branch (cc := 12) (yes := [.breakLoop]) (no := []) (following := factorIteration.drop 1 ++ continuation) (flag := false)
          (by simpa [stop] using stopTest) rfl branch
        exact (BlockReturnsIn.list_step (body := factorIteration) (otherwise := []) (following := factorTail) found item continued).mono (by omega)
  · rw [List.drop_eq_nil_of_le (by omega : ps.length ≤ position)]
    have run := BlockReturnsIn.list_done (name := "p") (body := factorIteration) (otherwise := []) (following := factorTail) found
      (List.getElem?_eq_none (by simpa [primeValues] using (show ps.length ≤ position by omega)))
      (factor_finish x total heap env getX getTotal)
    exact run.mono (by omega)
termination_by ps.length-position
decreasing_by all_goals omega

end solvers.«12».runtimeCost
