import solvers.«12.py.lean.d».SearchScalarRuntime

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

noncomputable def searchWork (k bound : Nat) : Nat :=
  2*boundedFactorBudget bound (ProjectEulerStatements.P12.firstIndex k+1)+120

private theorem finish_budget (cost count : Nat) (one : 1 ≤ count) :
    1+(30+((2*cost+56)+(7+17+2)+2))+2 ≤ (2*cost+120)*count := by
  have bound := Nat.mul_le_mul_left (2*cost+120) one
  simp only [Nat.mul_one] at bound
  omega

private theorem advance_budget (cost remaining : Nat) :
    1+(30+((2*cost+56)+(7+(7+(2*cost+120)*remaining)+2)+2))+2 ≤
      (2*cost+120)*(remaining+1) := by
  simp only [Nat.mul_add, Nat.mul_one]
  omega

/-- Every iteration executes both factor-count calls and preserves the bounded
cache until the first qualifying triangular index. -/
theorem search_loop (k n bound : Nat) (primeBound : Nat.Prime bound)
    (enough : Nat.sqrt (ProjectEulerStatements.P12.firstIndex k+1)+1 ≤ bound) (positive : 0 < n)
    (before : n ≤ Nat.find (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k))
    (heap : Mutable.Heap) (address storage : Nat) (cache : BoundedCache heap address storage bound)
    (env : HeapCalls.Env) (getN : env "n" = some (.scalar (.int n)))
    (getK : env "k" = some (.scalar (.int k))) (getTable : env "pt" = some (.ref address)) :
    ∃ finalHeap, BlockReturnsIn .word program env searchLoop heap
      (searchWork k bound*(ProjectEulerStatements.P12.firstIndex k+1-n))
      (.scalar (.int (ProjectEulerStatements.P12.firstTriangleOver k))) finalHeap := by
  let a := ProjectEulerStatements.P12.splitLeft n
  let b := ProjectEulerStatements.P12.splitRight n
  let t := ProjectEulerStatements.P12.triangle n
  let d := ProjectEulerStatements.P12.divisorCount t
  let parted := splitEnv env n
  let withD := HeapCalls.setLocal parted "d" (.scalar (.int d))
  let f := boundedFactorBudget bound (ProjectEulerStatements.P12.firstIndex k+1)
  have beforeIndex : n ≤ ProjectEulerStatements.P12.firstIndex k := before
  have splits := ProjectEulerStatements.P12.split_le_succ n
  have smallA : a ≤ ProjectEulerStatements.P12.firstIndex k+1 := by dsimp [a]; omega
  have smallB : b ≤ ProjectEulerStatements.P12.firstIndex k+1 := by dsimp [b]; omega
  have splitPositive := ProjectEulerStatements.P12.split_positive n positive
  obtain ⟨heap1, cache1, left⟩ := factor_call a (ProjectEulerStatements.P12.firstIndex k+1) bound splitPositive.1 smallA primeBound enough heap address storage cache parted "pt" (.name "a") 1
    (by simpa [parted, splitEnv, HeapCalls.setLocal] using getTable) (EvaluatesIn.lookup rfl)
  obtain ⟨heap2, cache2, right⟩ := factor_call b (ProjectEulerStatements.P12.firstIndex k+1) bound splitPositive.2 smallB primeBound enough heap1 address storage cache1 parted "pt" (.name "b") 1
    (by simpa [parted, splitEnv, HeapCalls.setLocal] using getTable) (EvaluatesIn.lookup rfl)
  have product : a.divisors.card * b.divisors.card = d :=
    (ProjectEulerStatements.P12.divisorCount_triangle_split n).symm
  have calculated : EvaluatesIn .word program parted
      (.binOp (.call (.attribute (.name "pt") "divisor_count") [.name "a"]) .mult
        (.call (.attribute (.name "pt") "divisor_count") [.name "b"])) heap (2*f+56) (.scalar (.int d)) heap2 := by
    apply EvaluatesIn.mono (small := (1+f+25)+(1+f+25)+1+3) (larger := by omega)
    apply EvaluatesIn.binary (work := 1) left right
    · simp [Mutable.binary, scalarBinary]
      exact_mod_cast product
    · rfl
  have tested : EvaluatesIn .word program withD (.compare (.name "d") [(.gt, .name "k")]) heap2 7
      (.scalar (.bool (decide (k < d)))) heap2 := by
    apply EvaluatesIn.compare (work := 2) (EvaluatesIn.lookup rfl)
      (EvaluatesIn.lookup (by simpa [withD, parted, splitEnv, HeapCalls.setLocal] using getK))
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
    · rfl
  by_cases qualifies : k < d
  · have least := Nat.find_min' (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k) qualifies
    have same : n = Nat.find (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k) := by omega
    refine ⟨heap2, ?_⟩
    have one : 1 ≤ ProjectEulerStatements.P12.firstIndex k+1-n := by omega
    apply BlockReturnsIn.mono (small := 1+(30+((2*f+56)+(7+17+2)+2))+2) (larger := finish_budget f _ one)
    apply BlockReturnsIn.while_loop (cc := 1) (cb := 30+((2*f+56)+(7+17+2)+2)) (flag := true) EvaluatesIn.constant rfl
    apply BlockContinuesIn.returns (ca := 30) (cb := (2*f+56)+(7+17+2)+2) (body := [searchBody[0]'(by decide)]) (split_cost n heap env getN)
    apply BlockReturnsIn.assign (ca := 2*f+56) (cb := 7+17+2) calculated
    apply BlockReturnsIn.branch (cc := 7) (cb := 17) (flag := true) (by simpa [qualifies] using tested) rfl
    apply BlockReturnsIn.return_value
    simpa only [ProjectEulerStatements.P12.firstTriangleOver, ← same] using triangle_cost n heap2 withD
      (by simpa [withD, parted, splitEnv, HeapCalls.setLocal] using getN)
  · have earlier : n < Nat.find (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k) := by
      have witness := Nat.find_spec (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k)
      by_contra late
      have same : n = Nat.find (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k) := by omega
      apply qualifies
      simpa only [d, t, same] using witness
    let advanced := HeapCalls.setLocal withD "n" (.scalar (.int (n + 1 : Nat)))
    obtain ⟨finalHeap, rest⟩ := search_loop k (n + 1) bound primeBound enough (by omega) (by omega) heap2 address storage cache2 advanced rfl
      (by simpa [advanced, withD, parted, splitEnv, HeapCalls.setLocal] using getK)
      (by simpa [advanced, withD, parted, splitEnv, HeapCalls.setLocal] using getTable)
    refine ⟨finalHeap, ?_⟩
    let remaining := searchWork k bound*(ProjectEulerStatements.P12.firstIndex k+1-(n+1))
    have earlierIndex : n < ProjectEulerStatements.P12.firstIndex k := earlier
    have measure : ProjectEulerStatements.P12.firstIndex k+1-n =
        (ProjectEulerStatements.P12.firstIndex k+1-(n+1))+1 := by omega
    apply BlockReturnsIn.mono (small := 1+(30+((2*f+56)+(7+(7+remaining)+2)+2))+2) (larger := by
      rw [measure]
      exact advance_budget f _)
    apply BlockReturnsIn.while_loop (cc := 1) (cb := 30+((2*f+56)+(7+(7+remaining)+2)+2)) (flag := true) EvaluatesIn.constant rfl
    apply BlockContinuesIn.returns (ca := 30) (cb := (2*f+56)+(7+(7+remaining)+2)+2) (body := [searchBody[0]'(by decide)]) (split_cost n heap env getN)
    apply BlockReturnsIn.assign (ca := 2*f+56) (cb := 7+(7+remaining)+2) calculated
    apply BlockReturnsIn.branch (cc := 7) (cb := 7+remaining) (flag := false) (by simpa [qualifies] using tested) rfl
    apply BlockReturnsIn.augAssign (work := 1) (ca := 1) (cb := remaining)
      (value := .scalar (.int (n + 1 : Nat))) (middle := heap2)
      (by simpa [withD, parted, splitEnv, HeapCalls.setLocal] using getN) EvaluatesIn.constant
    · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
    · rfl
    · exact rest
termination_by Nat.find (ProjectEulerStatements.P12.exists_triangle_divisorCount_gt k) - n
decreasing_by omega


end solvers.«12».runtimeCost
