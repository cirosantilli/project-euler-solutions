import solvers.«12.py.lean.d».FactorSource
import lang.py.FormalSW.HeapRuntimeCostSequences

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

/-- Each successful division pays from the decrease in the positive residue.
This budget can be shared across all primes in the outer factorization loop. -/
theorem strip_cost (x p exponent : Nat) (positive : 0 < x) (prime : Nat.Prime p)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getX : env "x" = some (.scalar (.int x)))
    (getP : env "p" = some (.scalar (.int p)))
    (getExponent : env "exp" = some (.scalar (.int exponent))) :
    ∃ finalEnv, BlockContinuesIn .word program env [stripLoop] heap
        (30*(x-(ProjectEulerStatements.FactorDivisors.strip x p).1)+14) finalEnv heap ∧
      finalEnv "x" = some (.scalar (.int (ProjectEulerStatements.FactorDivisors.strip x p).1)) ∧
      finalEnv "exp" = some (.scalar (.int (exponent + (ProjectEulerStatements.FactorDivisors.strip x p).2 : Nat))) ∧
      ∀ name, name ≠ "x" → name ≠ "exp" → finalEnv name = env name := by
  have tested : EvaluatesIn .word program env
      (.compare (.binOp (.name "x") .mod (.name "p")) [(.eq, .constant (.int 0))]) heap 12
      (.scalar (.bool (decide (p ∣ x)))) heap := by
    apply EvaluatesIn.compare (work := 2) (av := .scalar (.int (x % p))) (middle := heap)
      (EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup getX) (EvaluatesIn.lookup getP)
        (by simp [Mutable.binary, scalarBinary, prime.pos, prime.ne_zero]) rfl) EvaluatesIn.constant
    · simp [HeapCalls.compare, scalarCompare, evalCmpOp, ← Int.dvd_iff_emod_eq_zero, Int.natCast_dvd_natCast]
    · rfl
  by_cases divides : p ∣ x
  · let divided := HeapCalls.setLocal env "x" (.scalar (.int (x / p : Nat)))
    let advanced := HeapCalls.setLocal divided "exp" (.scalar (.int (exponent + 1 : Nat)))
    have first : BlockContinuesIn .word program env
        [.augAssign (.name "x") .floorDiv (.name "p")] heap 7 divided heap := by
      apply BlockContinuesIn.augAssign (work := 1) getX (EvaluatesIn.lookup getP)
      · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, prime.pos, prime.ne_zero]
      · rfl
    have second : BlockContinuesIn .word program divided
        [.augAssign (.name "exp") .add (.constant (.int 1))] heap 7 advanced heap := by
      apply BlockContinuesIn.augAssign (work := 1)
        (by simpa [divided, HeapCalls.setLocal] using getExponent) EvaluatesIn.constant
      · simp [HeapCalls.inplace, Mutable.binary, scalarBinary]
      · rfl
    have quotientPositive : 0 < x / p := Nat.div_pos (Nat.le_of_dvd positive divides) prime.pos
    obtain ⟨finalEnv, rest, finalX, finalExponent, preserved⟩ :=
      strip_cost (x / p) p (exponent + 1) quotientPositive prime heap advanced
        (by simp [advanced, divided, HeapCalls.setLocal])
        (by simpa [advanced, divided, HeapCalls.setLocal] using getP) rfl
    have stripEq : ProjectEulerStatements.FactorDivisors.strip x p =
        ((ProjectEulerStatements.FactorDivisors.strip (x / p) p).1,
          (ProjectEulerStatements.FactorDivisors.strip (x / p) p).2 + 1) := by
      rw [ProjectEulerStatements.FactorDivisors.strip, dif_pos ⟨prime.one_lt, positive, divides⟩]
    refine ⟨finalEnv, ?_, ?_, ?_, ?_⟩
    · have paid : BlockContinuesIn .word program env [stripLoop] heap
          (12+((7+7)+(30*(x/p-(ProjectEulerStatements.FactorDivisors.strip (x/p) p).1)+14))+2) finalEnv heap := by
        apply BlockContinuesIn.while_loop (cc := 12) (flag := true) (by simpa [divides] using tested) rfl
        exact (first.append second).append rest
      apply paid.mono
      have decreasing := Nat.div_lt_self positive prime.one_lt
      have residue := ProjectEulerStatements.FactorDivisors.strip_le (x/p) p quotientPositive prime
      simp only [stripEq]
      omega
    · simpa only [stripEq] using finalX
    · simpa only [stripEq, Nat.add_assoc, Nat.add_comm 1] using finalExponent
    · intro name notX notExponent
      rw [preserved name notX notExponent]
      simp [advanced, divided, HeapCalls.setLocal, notX, notExponent]
  · have stripEq : ProjectEulerStatements.FactorDivisors.strip x p = (x, 0) := by
      rw [ProjectEulerStatements.FactorDivisors.strip, dif_neg (by tauto)]
    refine ⟨env, ?_, ?_, ?_, by intros; rfl⟩
    · simp only [stripEq, Nat.sub_self, Nat.mul_zero, Nat.zero_add]
      exact BlockContinuesIn.while_loop (cc := 12) (cb := 0) (flag := false) (by simpa [divides] using tested) rfl
        BlockContinuesIn.nil
    · simpa only [stripEq] using getX
    · simpa only [stripEq, Nat.add_zero] using getExponent
termination_by x
decreasing_by exact Nat.div_lt_self positive prime.one_lt


end solvers.«12».runtimeCost
