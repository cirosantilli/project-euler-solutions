import solvers.«9.py.lean.d».GuardRuntime

namespace solvers.«9».runtimeCost
open FormalSW.PyAst HeapCalls.Runtime solvers.«9».runtimeSource
private abbrev program := Solvers.P9Ast.module

theorem coefficientA_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (k m n : Int)
    (hk : env "k" = some (.scalar (.int k)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    EvaluatesIn .word program env aExpression heap 21 (.scalar (.int (k * (m * m - n * n)))) heap :=
  integer_binary (EvaluatesIn.lookup hk)
    (integer_binary
      (integer_binary (EvaluatesIn.lookup hm) (EvaluatesIn.lookup hm) rfl)
      (integer_binary (EvaluatesIn.lookup hn) (EvaluatesIn.lookup hn) rfl) rfl) rfl

theorem coefficientB_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (k m n : Int)
    (hk : env "k" = some (.scalar (.int k)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    EvaluatesIn .word program env bExpression heap 16 (.scalar (.int (k * (2 * m * n)))) heap :=
  integer_binary (EvaluatesIn.lookup hk)
    (integer_binary
      (integer_binary EvaluatesIn.constant (EvaluatesIn.lookup hm) rfl)
      (EvaluatesIn.lookup hn) rfl) rfl

theorem coefficientC_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (k m n : Int)
    (hk : env "k" = some (.scalar (.int k)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n))) :
    EvaluatesIn .word program env cExpression heap 21 (.scalar (.int (k * (m * m + n * n)))) heap :=
  integer_binary (EvaluatesIn.lookup hk)
    (integer_binary
      (integer_binary (EvaluatesIn.lookup hm) (EvaluatesIn.lookup hm) rfl)
      (integer_binary (EvaluatesIn.lookup hn) (EvaluatesIn.lookup hn) rfl) rfl) rfl

theorem coefficientBody_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n s : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n)))
    (hs : env "s" = some (.scalar (.int s))) (positive : 0 < s) :
    BlockContinuesIn .word program env coefficientBody heap 72 (coefficientEnv env total m n s) heap := by
  have divide : EvaluatesIn .word program env kExpression heap 6 (.scalar (.int (total / s))) heap := by
    apply integer_binary (EvaluatesIn.lookup ht) (EvaluatesIn.lookup hs)
    simp [Mutable.binary, scalarBinary, positive, (show s ≠ 0 by omega)]
  apply BlockContinuesIn.append (cb := 64) (BlockContinuesIn.assign divide)
  apply BlockContinuesIn.append (cb := 41) (BlockContinuesIn.assign
    (coefficientA_cost _ heap (total / s) m n rfl
      (by simpa [HeapCalls.setLocal] using hm) (by simpa [HeapCalls.setLocal] using hn)))
  apply BlockContinuesIn.append (cb := 23) (BlockContinuesIn.assign
    (coefficientB_cost _ heap (total / s) m n rfl
      (by simpa [HeapCalls.setLocal] using hm) (by simpa [HeapCalls.setLocal] using hn)))
  exact BlockContinuesIn.assign (coefficientC_cost _ heap (total / s) m n rfl
    (by simpa [HeapCalls.setLocal] using hm) (by simpa [HeapCalls.setLocal] using hn))


theorem successBody_cost (env : HeapCalls.Env) (heap : Mutable.Heap) (total m n s best : Int)
    (ht : env "total" = some (.scalar (.int total)))
    (hm : env "m" = some (.scalar (.int m))) (hn : env "n" = some (.scalar (.int n)))
    (hs : env "s" = some (.scalar (.int s))) (hbest : env "best" = some (.scalar (.int best)))
    (positive : 0 < s) :
    ∃ newBest finalEnv finalHeap cost, cost ≤ 240 ∧
      BlockContinuesIn .word program env successBody heap cost finalEnv finalHeap ∧
      finalEnv "total" = some (.scalar (.int total)) ∧ finalEnv "m" = some (.scalar (.int m)) ∧
      finalEnv "best" = some (.scalar (.int newBest)) := by
  have coefficients := coefficientBody_cost env heap total m n s ht hm hn hs positive
  obtain ⟨x, y, z, sorted⟩ := sorted_any_cost (coefficientEnv env total m n s) heap
    (total / s * (m * m - n * n)) (total / s * (2 * m * n)) (total / s * (m * m + n * n)) rfl rfl rfl
  let sortedEnv := HeapCalls.assignNames (coefficientEnv env total m n s) ["a", "b", "c"] (triple x y z)
  have keptT : sortedEnv "total" = some (.scalar (.int total)) := by
    simpa [sortedEnv, coefficientEnv, HeapCalls.assignNames, triple, HeapCalls.setLocal] using ht
  have keptM : sortedEnv "m" = some (.scalar (.int m)) := by
    simpa [sortedEnv, coefficientEnv, HeapCalls.assignNames, triple, HeapCalls.setLocal] using hm
  have keptBest : sortedEnv "best" = some (.scalar (.int best)) := by
    simpa [sortedEnv, coefficientEnv, HeapCalls.assignNames, triple, HeapCalls.setLocal] using hbest
  obtain ⟨newBest, finalEnv, cc, bounded, updated, finalT, finalM, finalBest⟩ :=
    updateBest_cost sortedEnv _ total x y z best keptT rfl rfl rfl keptBest
  refine ⟨newBest, finalEnv, _, (72 + 72) + cc, by omega, (coefficients.append sorted).append updated,
    finalT, ?_, finalBest⟩
  exact finalM.trans keptM



end solvers.«9».runtimeCost
