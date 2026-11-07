import solvers.«9.py.lean.d».CandidateRuntime
import lang.py.FormalSW.HeapRuntimeCostLoopControl

namespace solvers.«9».runtimeCost
open FormalSW.PyAst HeapCalls.Runtime solvers.«9».runtimeSource
private abbrev program := Solvers.P9Ast.module

theorem sExpression_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (m n : Int)
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    EvaluatesIn .word program env sExpression heap 16 (.scalar (.int (2 * m * (m + n)))) heap :=
  integer_binary
    (integer_binary EvaluatesIn.constant (EvaluatesIn.lookup hm) rfl)
    (integer_binary (EvaluatesIn.lookup hm) (EvaluatesIn.lookup hn) rfl) rfl

theorem skipCondition_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total s : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (hs : env "s" = some (.scalar (.int s))) (positive : 0 < s) :
    EvaluatesIn .word program env skipCondition heap 12 (.scalar (.bool (decide (total % s ≠ 0)))) heap := by
  have remainder : EvaluatesIn .word program env (.binOp (.name "total") .mod (.name "s"))
      heap 6 (.scalar (.int (total % s))) heap := by
    apply integer_binary (EvaluatesIn.lookup ht) (EvaluatesIn.lookup hs)
    simp [Mutable.binary, scalarBinary, positive, (show s ≠ 0 by omega)]
  apply integer_compare remainder EvaluatesIn.constant
  simp [HeapCalls.compare, scalarCompare, evalCmpOp]
/-- Runtime of the actual inner range, including both continue and sort paths. -/
theorem innerRange_cost (total best : Int) (m n : Nat) (positive : 0 < n)
    (env : HeapCalls.Env) (heap : Mutable.Heap)
    (ht : env "total" = some (.scalar (.int total))) (hm : env "m" = some (.scalar (.int m)))
    (hbest : env "best" = some (.scalar (.int best))) :
    ∃ newBest finalEnv finalHeap cost, cost ≤ 300 * (m - n) + 4 ∧
      BlockContinuesIn .word program env [.rangeStepLoop "n" n m 1 innerBody []] heap cost finalEnv finalHeap ∧
      finalEnv "total" = some (.scalar (.int total)) ∧ finalEnv "m" = some (.scalar (.int m)) ∧
      finalEnv "best" = some (.scalar (.int newBest)) := by
  by_cases before : n < m
  · let denominator : Int := 2 * (m : Int) * ((m : Int) + n)
    let nextEnv := HeapCalls.setLocal env "n" (.scalar (.int n))
    let withS := HeapCalls.setLocal nextEnv "s" (.scalar (.int denominator))
    have nt : nextEnv "total" = some (.scalar (.int total)) := by simpa [nextEnv, HeapCalls.setLocal] using ht
    have nm : nextEnv "m" = some (.scalar (.int m)) := by simpa [nextEnv, HeapCalls.setLocal] using hm
    have st : withS "total" = some (.scalar (.int total)) := by simpa [withS, HeapCalls.setLocal] using nt
    have sm : withS "m" = some (.scalar (.int m)) := by simpa [withS, HeapCalls.setLocal] using nm
    have sbest : withS "best" = some (.scalar (.int best)) := by simpa [withS, nextEnv, HeapCalls.setLocal] using hbest
    have denominatorPositive : 0 < denominator :=
      Int.mul_pos (Int.mul_pos (by decide) (by omega)) (by omega)
    have assigned := BlockContinuesIn.assign (name := "s") (sExpression_cost nextEnv heap m n nm rfl)
    have guard := skipCondition_cost withS heap total denominator st rfl denominatorPositive
    have running : Calls.rangeContinues (n : Int) m 1 = true := by simp [Calls.rangeContinues, before]
    by_cases divides : total % denominator = 0
    · obtain ⟨rowBest, rowEnv, rowHeap, successCost, successBound, success, rowT, rowM, rowB⟩ :=
        successBody_cost withS heap total m n denominator best st sm rfl rfl sbest denominatorPositive
      obtain ⟨newBest, finalEnv, finalHeap, restCost, restBound, rest, keptT, keptM, keptB⟩ :=
        innerRange_cost total rowBest m (n + 1) (by omega) rowEnv rowHeap rowT rowM rowB
      refine ⟨newBest, finalEnv, finalHeap,
        (18 + (12 + (successCost + (restCost + 1)) + 2)) + 4, by omega, ?_, keptT, keptM, keptB⟩
      apply BlockContinuesIn.range_step running
      apply BlockContinuesIn.append assigned
      apply BlockContinuesIn.branch_tail (flag := false) (value := .bool false)
      · simpa [divides] using guard
      · rfl
      · simpa only [Int.natCast_add, Int.natCast_one] using
          success.append (BlockContinuesIn.loop_resume rest)
    · obtain ⟨newBest, finalEnv, finalHeap, restCost, restBound, rest, keptT, keptM, keptB⟩ :=
        innerRange_cost total best m (n + 1) (by omega) withS heap st sm sbest
      refine ⟨newBest, finalEnv, finalHeap,
        (18 + (12 + (restCost + 1) + 2)) + 4, by omega, ?_, keptT, keptM, keptB⟩
      apply BlockContinuesIn.range_step running
      apply BlockContinuesIn.append assigned
      apply BlockContinuesIn.branch_tail (flag := true) (value := .bool true)
      · simpa [divides] using guard
      · rfl
      · apply BlockContinuesIn.continue_over (by intro following; rfl)
        simpa only [Int.natCast_add, Int.natCast_one] using rest
  · refine ⟨best, env, heap, 4, by omega,
      BlockContinuesIn.range_done ?_ BlockContinuesIn.nil, ht, hm, hbest⟩
    simp [Calls.rangeContinues, before]
termination_by m - n
decreasing_by all_goals omega

end solvers.«9».runtimeCost
