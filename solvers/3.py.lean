import «.formalsw».solvers.«3.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p3.py»
import FormalSW
import ProjectEulerSolutions.Equiv.P3

namespace solvers.«3»
open FormalSW.PyAst FormalSW.PyAst.Calls
open ProjectEulerStatements.P3 ProjectEulerSolutions.P3

private abbrev program := Solvers.P3Ast.module
private abbrev answer (n : Nat) := Nat.findGreatest (isPrimeFactor n) n

private structure Invariant (N n f largest : Nat) : Prop where
  input : 2 ≤ N
  divides : n ∣ N
  positive : 0 < n
  largest_ok : LargestOK N largest
  cover : Cover N n largest
  low : Low n f
  factor : 3 ≤ f
  odd : Odd f
  largest_le : largest ≤ f
  remaining : n > 1 → largest ≤ n
  last : n ≤ 1 → largest ≠ 1

private theorem Invariant.divide (h : Invariant N n f largest) (hd : f ∣ n) :
    Invariant N (n / f) f f := by
  have fp : Nat.Prime f := prime_of_divides_with_low (by have := h.factor; omega) hd h.low
  have low : Low (n / f) f := fun p hp hd' => h.low p hp
    (dvd_trans hd' (Nat.div_dvd_of_dvd hd))
  refine ⟨h.input, dvd_trans (Nat.div_dvd_of_dvd hd) h.divides,
    Nat.div_pos (Nat.le_of_dvd h.positive hd) fp.pos,
    Or.inr ⟨fp.two_le, dvd_trans hd h.divides, fp⟩, ?_, low,
    h.factor, h.odd, le_rfl, ?_, ?_⟩
  · intro p hp
    rcases h.cover p hp with bound | divisor
    · exact Or.inl (le_trans bound h.largest_le)
    · by_cases bound : p ≤ f
      · exact Or.inl bound
      · exact Or.inr (prime_dvd_div_of_ne hd fp hp.2.2 divisor (by omega))
  · intro more
    obtain ⟨p, hp, hd'⟩ := Nat.exists_prime_and_dvd (by omega : n / f ≠ 1)
    exact le_trans (low p hp hd') (Nat.le_of_dvd (by omega) hd')
  · intro _
    exact fp.ne_one

private theorem Invariant.advance (h : Invariant N n f largest) (hd : ¬ f ∣ n) :
    Invariant N n (f + 2) largest := by
  refine ⟨h.input, h.divides, h.positive, h.largest_ok, h.cover,
    lower_step h.factor h.odd h.low hd, by have := h.factor; omega,
    ?_, by have := h.largest_le; omega, h.remaining, h.last⟩
  obtain ⟨k, hk⟩ := h.odd
  exact ⟨k + 1, by omega⟩

private def finish : List Stmt :=
  [.ifThen (.compare (.name "n") [(.gt, .constant (.int 1))])
    [.assign [.name "largest"] (.name "n")] [], .return (.name "largest")]

private def modTest (factor : Expr) : Expr :=
  .compare (.binOp (.name "n") .mod factor) [(.eq, .constant (.int 0))]

private def stripBody : List Stmt :=
  [.assign [.name "largest"] (.name "f"), .augAssign (.name "n") .floorDiv (.name "f")]

private def outerTest : Expr :=
  .compare (.binOp (.name "f") .mult (.name "f")) [(.ltE, .name "n")]

private def outerBody : List Stmt :=
  [.whileDo (modTest (.name "f")) stripBody [], .augAssign (.name "f") .add (.constant (.int 2))]

private def outerLoop : List Stmt := .whileDo outerTest outerBody [] :: finish

private theorem mod_result (n f : Nat) (hf : 0 < f) (env : ScalarEnv) (expr : Expr)
    (hn : env "n" = some (.int n)) (he : Evaluates program env expr (.int f)) :
    Evaluates program env (modTest expr) (.bool (decide (f ∣ n))) := by
  apply Evaluates.compare (av := .int ↑(n % f)) (bv := .int 0)
  · exact Evaluates.binary (Evaluates.lookup hn) he (by simp [scalarBinary, hf.ne'])
  · exact Evaluates.constant
  · simp [scalarCompare, evalCmpOp, Int.natCast_dvd_natCast]

private theorem finish_result (h : Invariant N n f largest) (stopped : ¬ f * f ≤ n)
    (env : ScalarEnv) (hn : env "n" = some (.int n))
    (hl : env "largest" = some (.int largest)) :
    BlockReturns program env finish (.int (answer N)) := by
  change BlockReturns program env finish (.int (Nat.findGreatest (isPrimeFactor N) N))
  by_cases more : n > 1
  · have result := result_of_state_final_large h.divides h.largest_ok h.cover h.low h.remaining more stopped
    rw [resultOK_findGreatest (by have := h.input; omega) result]
    apply BlockReturns.branch (value := .bool true)
    · exact Evaluates.compare (Evaluates.lookup hn) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp]; omega)
    · apply BlockReturns.assign (Evaluates.lookup hn)
      exact BlockReturns.return_value (Evaluates.lookup (by simp [ScalarEnv.set]))
  · have result := result_of_state_final_small h.input h.largest_ok h.cover h.positive more h.last
    rw [resultOK_findGreatest (by have := h.input; omega) result]
    apply BlockReturns.branch (value := .bool false)
    · exact Evaluates.compare (Evaluates.lookup hn) Evaluates.constant
        (by simp [scalarCompare, evalCmpOp]; omega)
    · exact BlockReturns.return_value (Evaluates.lookup hl)

private theorem strip_result (N n f largest : Nat) (h : Invariant N n f largest)
    (env : ScalarEnv) (hn : env "n" = some (.int n))
    (hf : env "f" = some (.int f)) (hl : env "largest" = some (.int largest))
    (following : List Stmt)
    (continuation : ∀ n' largest' env', n' ≤ n → Invariant N n' f largest' → ¬ f ∣ n' →
      env' "n" = some (.int n') → env' "f" = some (.int f) →
      env' "largest" = some (.int largest') →
      BlockReturns program env' following (.int (answer N))) :
    BlockReturns program env (.whileDo (modTest (.name "f")) stripBody [] :: following)
      (.int (answer N)) := by
  induction n using Nat.strong_induction_on generalizing largest env with
  | h n ih =>
      by_cases hd : f ∣ n
      · apply BlockReturns.while_loop (value := .bool true)
        · simpa [hd] using mod_result n f (by have := h.factor; omega) env _ hn (Evaluates.lookup hf)
        · apply BlockReturns.assign (Evaluates.lookup hf)
          apply BlockReturns.augAssign (value := .int ↑(n / f))
          · exact Evaluates.binary (Evaluates.lookup (by simpa [ScalarEnv.set] using hn))
              (Evaluates.lookup (by simpa [ScalarEnv.set] using hf))
              (by simp [scalarBinary, show f ≠ 0 by have := h.factor; omega])
          · apply ih (n / f) (Nat.div_lt_self h.positive (by have := h.factor; omega))
              f (h.divide hd)
            · simp [ScalarEnv.set]
            · simpa [ScalarEnv.set] using hf
            · simp [ScalarEnv.set]
            · intro n' largest' env' bound inv indiv en ef el
              exact continuation n' largest' env' (le_trans bound (Nat.div_le_self _ _)) inv indiv en ef el
      · apply BlockReturns.while_loop (value := .bool false)
        · simpa [hd] using mod_result n f (by have := h.factor; omega) env _ hn (Evaluates.lookup hf)
        · exact continuation n largest env le_rfl h hd hn hf hl

private theorem outer_result (N n f largest : Nat) (h : Invariant N n f largest)
    (env : ScalarEnv) (hn : env "n" = some (.int n))
    (hf : env "f" = some (.int f)) (hl : env "largest" = some (.int largest)) :
    BlockReturns program env outerLoop (.int (answer N)) := by
  induction n using Nat.strong_induction_on generalizing f largest env with
  | h n smaller_n =>
      induction measure : n - f using Nat.strong_induction_on generalizing f largest env with
      | h size smaller_f =>
          have test : Evaluates program env outerTest (.bool (decide (f * f ≤ n))) := by
            exact Evaluates.compare (av := .int ↑(f * f)) (bv := .int n)
              (Evaluates.binary (Evaluates.lookup hf) (Evaluates.lookup hf)
              (by simp [scalarBinary])) (Evaluates.lookup hn)
              (by simp [scalarCompare, evalCmpOp]; norm_cast)
          by_cases more : f * f ≤ n
          · apply BlockReturns.while_loop (value := .bool true)
            · simpa [more] using test
            · apply strip_result N n f largest h env hn hf hl
              intro n' largest' env' bound inv indiv en ef el
              apply BlockReturns.augAssign (value := .int ↑(f + 2))
              · exact Evaluates.binary (Evaluates.lookup ef) Evaluates.constant (by simp [scalarBinary])
              · have next := inv.advance indiv
                by_cases reduced : n' < n
                · apply smaller_n n' reduced (f + 2) largest' next
                  · simpa [ScalarEnv.set] using en
                  · simp [ScalarEnv.set]
                  · simpa [ScalarEnv.set] using el
                · have same : n' = n := by omega
                  subst n'
                  have decreases : n - (f + 2) < size := by
                    have factor := h.factor
                    have enough : f + 2 ≤ n := by nlinarith
                    omega
                  apply smaller_f (n - (f + 2)) decreases (f + 2) largest' next
                  · simpa [ScalarEnv.set] using en
                  · simp [ScalarEnv.set]
                  · simpa [ScalarEnv.set] using el
                  · rfl
          · apply BlockReturns.while_loop (value := .bool false)
            · simpa [more] using test
            · exact finish_result h more env hn hl

private def twosBody : List Stmt :=
  [.assign [.name "largest"] (.constant (.int 2)), .augAssign (.name "n") .floorDiv (.constant (.int 2))]

private theorem twos_result (n largest : Nat) (positive : 0 < n) (env : ScalarEnv)
    (hn : env "n" = some (.int n)) (hl : env "largest" = some (.int largest))
    (following : List Stmt) (value : Constant)
    (continuation : ∀ env', env' "n" = some (.int (stripTwos n largest).1) →
      env' "largest" = some (.int (stripTwos n largest).2) → BlockReturns program env' following value) :
    BlockReturns program env (.whileDo (modTest (.constant (.int 2))) twosBody [] :: following) value := by
  induction n using Nat.strong_induction_on generalizing largest env with
  | h n ih =>
      by_cases even : 2 ∣ n
      · have unfold_strip : stripTwos n largest = stripTwos (n / 2) 2 := by
          rw [stripTwos.eq_def]
          simp [positive.ne', Nat.mod_eq_zero_of_dvd even]
        apply BlockReturns.while_loop (value := .bool true)
        · simpa [even] using mod_result n 2 (by decide) env _ hn Evaluates.constant
        · apply BlockReturns.assign Evaluates.constant
          apply BlockReturns.augAssign (value := .int ↑(n / 2))
          · exact Evaluates.binary (Evaluates.lookup (by simpa [ScalarEnv.set] using hn))
              Evaluates.constant (by simp [scalarBinary])
          · apply ih (n / 2) (Nat.div_lt_self positive (by decide)) 2
              (Nat.div_pos (Nat.le_of_dvd positive even) (by decide))
            · simp [ScalarEnv.set]
            · simp [ScalarEnv.set]
            · simpa only [unfold_strip] using continuation
      · have unfold_strip : stripTwos n largest = (n, largest) := by
          rw [stripTwos.eq_def]
          simp [positive.ne', show n % 2 ≠ 0 from fun eq => even (Nat.dvd_of_mod_eq_zero eq)]
        apply BlockReturns.while_loop (value := .bool false)
        · simpa [even] using mod_result n 2 (by decide) env _ hn Evaluates.constant
        · exact continuation env (by simpa [unfold_strip] using hn) (by simpa [unfold_strip] using hl)

theorem solve.result (N : Nat) (valid : 2 ≤ N) :
    ReturnsValue program "solve" [.int N] (.int (answer N)) := by
  have positive : 0 < N := by omega
  have strip : StripOK N (stripTwos N 1) := by
    apply stripTwos_correct dvd_rfl positive (Or.inl rfl)
    · exact fun p hp => Or.inr hp.2.1
    · decide
    · omega
    · omega
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign Evaluates.constant
  apply twos_result N 1 positive _ rfl rfl
  intro env hn hl
  apply BlockReturns.assign Evaluates.constant
  apply outer_result N (stripTwos N 1).1 3 (stripTwos N 1).2
  · exact ⟨valid, strip.1, strip.2.1, strip.2.2.1, strip.2.2.2.1,
      strip.2.2.2.2.1, by decide, by decide, strip.2.2.2.2.2.1,
      strip.2.2.2.2.2.2.1, strip.2.2.2.2.2.2.2⟩
  · simpa [ScalarEnv.set] using hn
  · simp [ScalarEnv.set]
  · simpa [ScalarEnv.set] using hl

theorem solve.«Equiv(naive)» (n : Int) (hp : Calls.Condition program
      (fun key => if key = "n" then some (.int n) else none) Solvers.P3Ast.requires_solve_0) :
    (Calls.function program "solve" [.int n]).Equiv
      (Calls.function P3Ast.module "naive" [.int n]) := by
  have hn : n ≥ 2 := by
    simpa [evalCmpOp] using Evaluates.condition_int_compare hp rfl
  lift n to Nat using (by omega : 0 ≤ n)
  exact (solve.result n (by omega)).equiv
    (data.«project-euler-statements».data.py.project_euler_statements.p3.naive.result n)

end solvers.«3»
