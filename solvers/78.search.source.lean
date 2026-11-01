import solvers.«78.iteration.source»
import ProjectEulerStatements.Lib.PartitionSearch
import lang.py.FormalSW.LoopProgress

/-! The mutable residue table searches precisely the positive divisible partition indices. -/
namespace solvers.«78».solve
open FormalSW.PyAst HeapCalls
open ProjectEulerStatements.PartitionSearch ProjectEulerStatements.PartitionRecurrence
set_option Elab.async false

private abbrev statement : Stmt := .whileDo (.constant (.bool true)) inner.solveBody []

private theorem tested (modulus : Int) (n : Nat) (heap : Heap) (env : HeapCalls.Env)
    (getTotal : env "total" = some (number (count n % modulus))) :
    Evaluates program env (.compare (.name "total") [(.eq, .constant (.int 0))]) heap
      (flag (decide (count n % modulus = 0))) heap := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (finalHeap := heap)
    (Evaluates.lookup getTotal) Evaluates.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]

private theorem scan_result (modulus : Int) (positive : 0 < modulus) (success : Success modulus)
    (n : Nat) (values : List Int) (valid : Valid modulus n values) (base : Heap)
    (env : HeapCalls.Env) (lower : 0 < n) (upper : n ≤ answer modulus success)
    (getN : env "n" = some (number n)) (getMod : env "mod" = some (number modulus))
    (getP : env "p" = some (.ref base.length)) :
    ∃ finalValues, BlockReturns program env [statement] (working base values)
      (number (answer modulus success)) (working base finalValues) := by
  induction measure : answer modulus success-n using Nat.strong_induction_on generalizing n values env with
  | h size ih =>
      obtain ⟨middleEnv, processed, getTotal, kept⟩ :=
        iteration_result modulus positive n lower values valid base env getN getMod getP
      have middleN := (kept "n" (by decide)).trans getN
      have middleMod := (kept "mod" (by decide)).trans getMod
      have middleP := (kept "p" (by decide)).trans getP
      have test := tested modulus n (working base (values ++ [count n % modulus])) middleEnv getTotal
      by_cases same : n = answer modulus success
      · refine ⟨values ++ [count n % modulus], ?_⟩
        apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
        change BlockReturns program env
          (inner.solveBody.take 5 ++ (inner.solveBody.drop 5 ++ [statement]))
          (working base values) _ _
        apply processed.returns
        have divides : count n % modulus = 0 := by
          simpa only [same, count] using (answer_spec modulus success).2
        have truth : decide (count n % modulus = 0) = true := by simp [divides]
        apply BlockReturns.branch (flag := true)
          (by simpa only [truth] using test) rfl
        have finalN : middleEnv "n" = some (number (answer modulus success)) := by
          simpa only [same] using middleN
        exact BlockReturns.return_value
          (Evaluates.lookup finalN)
      · have earlier : n < answer modulus success := by omega
        have missing : count n % modulus ≠ 0 := before_answer modulus success n lower earlier
        let advanced := setLocal middleEnv "n" (number (n+1 : Nat))
        obtain ⟨finalValues, rest⟩ := ih (answer modulus success-(n+1)) (by omega) (n+1)
          (values ++ [count n % modulus]) (valid_append modulus n values valid) advanced
          (by omega) (by omega) (by simp [advanced, setLocal])
          (by simpa [advanced, setLocal] using middleMod)
          (by simpa [advanced, setLocal] using middleP) rfl
        refine ⟨finalValues, ?_⟩
        apply BlockReturns.while_loop (flag := true) Evaluates.constant rfl
        change BlockReturns program env
          (inner.solveBody.take 5 ++ (inner.solveBody.drop 5 ++ [statement]))
          (working base values) _ _
        apply processed.returns
        apply BlockReturns.branch (flag := false)
          (by simpa only [missing, decide_false] using test) rfl
        apply BlockReturns.augAssign (left := number n) (right := number 1)
          (value := number (n+1 : Nat)) (afterOperation := working base (values ++ [count n % modulus]))
          middleN Evaluates.constant
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
        · exact rest

theorem result (modulus : Int) (positive : 0 < modulus) (success : Success modulus) (heap : Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number modulus] heap
      (number (answer modulus success)) finalHeap := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let entry := functionEnv args body [number modulus] []
  let withTable := setLocal entry "p" (.ref heap.length)
  let ready := setLocal withTable "n" (number 1)
  obtain ⟨finalValues, finished⟩ := scan_result modulus positive success 1 [1] (valid_initial modulus)
    heap ready (by omega) (answer_positive modulus success) (by simp [ready, setLocal])
    (by simp [ready, withTable, setLocal]; rfl) (by simp [ready, withTable, setLocal])
  refine ⟨working heap finalValues, ?_⟩
  apply BlockReturns.function (args := args) (body := body) rfl rfl
  apply (BlockContinues.annAssign (Evaluates.list (.cons Evaluates.constant .nil))).returns
  apply BlockReturns.assign Evaluates.constant
  exact finished

theorem diverges (modulus : Int) (positive : 0 < modulus) (none : ¬ Success modulus) :
    (HeapCalls.scalarFunction program "solve" [.int modulus]).Diverges := by
  let args := ((findFunctionIn program.body "solve").getD ([], [])).1
  let body := ((findFunctionIn program.body "solve").getD ([], [])).2
  let entry := functionEnv args body [number modulus] []
  let ready := setLocal (setLocal entry "p" (.ref 0)) "n" (number 1)
  let invariant : HeapCalls.Env → Heap → Prop := fun env heap =>
    ∃ n : Nat, ∃ values : List Int, 0 < n ∧ Valid modulus n values ∧ heap = working [] values ∧
      env "n" = some (number n) ∧ env "mod" = some (number modulus) ∧ env "p" = some (.ref 0)
  apply scalar_while_diverges program "solve" [.int modulus] args body (inner.solveCode.take 2)
    inner.solveBody [number modulus] rfl rfl rfl rfl invariant ready (working [] [1])
  · exact (BlockContinues.annAssign (Evaluates.list (.cons Evaluates.constant .nil))).append
      (BlockContinues.assign Evaluates.constant)
  · exact ⟨1, [1], by omega, valid_initial modulus, rfl, by simp [ready, setLocal],
      by simp [ready, setLocal]; rfl, by simp [ready, setLocal]⟩
  · intro env heap current
    obtain ⟨n, values, lower, valid, rfl, getN, getMod, getP⟩ := current
    obtain ⟨middleEnv, processed, getTotal, kept⟩ :=
      iteration_result modulus positive n lower values valid [] env getN getMod getP
    have middleN := (kept "n" (by decide)).trans getN
    have middleMod := (kept "mod" (by decide)).trans getMod
    have middleP := (kept "p" (by decide)).trans getP
    have missing : count n % modulus ≠ 0 := absent modulus none n lower
    have test := tested modulus n (working [] (values ++ [count n % modulus])) middleEnv getTotal
    have skipped : BlockContinues program middleEnv ((inner.solveBody.drop 5).take 1)
        (working [] (values ++ [count n % modulus])) middleEnv
        (working [] (values ++ [count n % modulus])) :=
      BlockContinues.branch (flag := false)
        (by simpa only [missing, decide_false] using test) rfl .nil
    let advanced := setLocal middleEnv "n" (number (n+1 : Nat))
    have increment : BlockContinues program middleEnv
        [.augAssign (.name "n") .add (.constant (.int 1))]
        (working [] (values ++ [count n % modulus])) advanced
        (working [] (values ++ [count n % modulus])) := by
      apply BlockContinues.augAssign middleN Evaluates.constant
      simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
    exact ⟨advanced, working [] (values ++ [count n % modulus]), processed.append (skipped.append increment),
      n+1, values ++ [count n % modulus], by omega, valid_append modulus n values valid, rfl,
      by simp [advanced, setLocal], by simpa [advanced, setLocal] using middleMod,
      by simpa [advanced, setLocal] using middleP⟩

theorem returns_iff (modulus : Int) (positive : 0 < modulus) :
    (HeapCalls.scalarFunction program "solve" [.int modulus]).Returns ↔ Success modulus := by
  classical
  constructor
  · rintro ⟨value, returned⟩
    by_contra none
    exact Execution.diverges_not_halts (diverges modulus positive none) ⟨.ok value, returned⟩
  · intro success
    obtain ⟨_, returned⟩ := result modulus positive success []
    exact returned.scalar_returns (values := [.int modulus])

end solvers.«78».solve
