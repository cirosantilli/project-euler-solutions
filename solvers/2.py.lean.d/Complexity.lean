import «.formalsw».solvers.«2.py.ast»
import data.«project-euler-statements».«.formalsw».data.py.project_euler_statements.«p2.py.ast»
import lang.py.FormalSW.CallsCostLoops

/-! Source-linked logarithmic execution bounds for both Euler 2 loops.
The size is the numeric input bound, not its bit length. Scalar integer operations
are atomic. Two actual Fibonacci updates at least double the first state value. -/
namespace solvers.«2».complexity
open FormalSW.PyAst FormalSW.PyAst.Calls
set_option maxRecDepth 4096
set_option maxHeartbeats 800000

private def program (optimized : Bool) : Module :=
  if optimized then Solvers.P2Ast.module else P2Ast.module
private def functionName (optimized : Bool) := if optimized then "solve" else "naive"
private def limitName (optimized : Bool) := if optimized then "limit" else "n"
private def termName (optimized : Bool) := if optimized then "a" else "b"
private def term (optimized : Bool) (a b : Nat) := if optimized then a else b
private def loop (optimized : Bool) : List Stmt :=
  (executableBody ((findFunctionIn (program optimized).body (functionName optimized)).getD ([], [])).2).drop
    (if optimized then 2 else 3)
private def whileBody (optimized : Bool) : List Stmt :=
  match loop optimized with
  | .whileDo _ body _ :: _ => body
  | _ => []
private def iterationBody (optimized : Bool) : List Stmt :=
  (whileBody optimized).drop (if optimized then 0 else 1)
private def afterGuard (optimized : Bool) := iterationBody optimized ++ loop optimized
private def afterParity (optimized : Bool) := (iterationBody optimized).drop 1 ++ loop optimized

private def Locals (optimized : Bool) (n a b total : Nat) (env : ScalarEnv) : Prop :=
  env (limitName optimized) = some (.int n) ∧ env "a" = some (.int a) ∧
  env "b" = some (.int b) ∧ env "total" = some (.int total)

private theorem term_lookup (mode : Bool) (n a b t : Nat) (env : ScalarEnv)
    (valid : Locals mode n a b t env) :
    env (termName mode) = some (.int (term mode a b)) := by
  cases mode with
  | false => exact valid.2.2.1
  | true => exact valid.2.1

private theorem guard_continue (mode : Bool) (n a b t : Nat) (env : ScalarEnv)
    (valid : Locals mode n a b t env) (active : term mode a b ≤ n) :
    ReachesIn (step (program mode)) ⟨.block (loop mode) env, []⟩
      (if mode then 7 else 10) ⟨.block (afterGuard mode) env, []⟩ := by
  obtain ⟨hn, ha, hb, ht⟩ := valid
  cases mode with
  | true =>
      have test : Cost.EvaluatesIn (program true) env
          (.compare (.name "a") [(.ltE, .name "limit")]) 5 (.bool true) :=
        Cost.EvaluatesIn.compare (Cost.EvaluatesIn.lookup ha) (Cost.EvaluatesIn.lookup hn)
          (by simp [scalarCompare, evalCmpOp]; exact active)
      exact Cost.whileTest test (whileBody true) [] [.return (.name "total")] []
  | false =>
      have test : Cost.EvaluatesIn (program false) env
          (.compare (.name "b") [(.gt, .name "n")]) 5 (.bool false) :=
        Cost.EvaluatesIn.compare (Cost.EvaluatesIn.lookup hb) (Cost.EvaluatesIn.lookup hn)
          (by simp [scalarCompare, evalCmpOp]; exact active)
      have lead := Cost.whileTest (module := program false) (env := env)
        (test := true) Cost.EvaluatesIn.constant (whileBody false) [] [] []
      have branch := Cost.branch test [.return (.name "total")] [] (afterGuard false) []
      exact lead.trans branch

private theorem guard_exit (mode : Bool) (n a b t : Nat) (env : ScalarEnv)
    (valid : Locals mode n a b t env) (finished : n < term mode a b) :
    FinishesIn (step (program mode)) ⟨.block (loop mode) env, []⟩
      (if mode then 10 else 13) (.ok (.int t)) := by
  obtain ⟨hn, ha, hb, ht⟩ := valid
  have tail (rest : List Stmt) : FinishesIn (step (program mode))
      ⟨.block (.return (.name "total") :: rest) env, []⟩ 3 (.ok (.int t)) :=
    .next rfl (.next (following := ⟨.value (.int t), []⟩) (by simp [step, ht, outcome]) (.done rfl))
  cases mode with
  | true =>
      have test : Cost.EvaluatesIn (program true) env
          (.compare (.name "a") [(.ltE, .name "limit")]) 5 (.bool false) :=
        Cost.EvaluatesIn.compare (Cost.EvaluatesIn.lookup ha) (Cost.EvaluatesIn.lookup hn)
          (by simp [scalarCompare, evalCmpOp]; exact finished)
      exact (Cost.whileTest test (whileBody true) [] [.return (.name "total")] []).finishes (tail [])
  | false =>
      have test : Cost.EvaluatesIn (program false) env
          (.compare (.name "b") [(.gt, .name "n")]) 5 (.bool true) :=
        Cost.EvaluatesIn.compare (Cost.EvaluatesIn.lookup hb) (Cost.EvaluatesIn.lookup hn)
          (by simp [scalarCompare, evalCmpOp]; exact finished)
      have lead := Cost.whileTest (module := program false) (env := env)
        (test := true) Cost.EvaluatesIn.constant (whileBody false) [] [] []
      have branch := Cost.branch test [.return (.name "total")] [] (afterGuard false) []
      exact (lead.trans branch).finishes (tail (afterGuard false))

private theorem unpack_cost (mode : Bool) (n a b t : Nat) (env : ScalarEnv)
    (valid : Locals mode n a b t env) :
    ∃ env', Locals mode n b (a + b) t env' ∧
      ReachesIn (step (program mode)) ⟨.block (afterParity mode) env, []⟩ 9
        ⟨.block (loop mode) env', []⟩ := by
  obtain ⟨hn, ha, hb, ht⟩ := valid
  let env' := (env.set "a" (.int b)).set "b" (.int (a + b))
  refine ⟨env', ?_, ?_⟩
  · cases mode <;> simpa [Locals, limitName, env', ScalarEnv.set] using And.intro hn ht
  · have right : Cost.EvaluatesIn (program mode) env
        (.binOp (.name "a") .add (.name "b")) 5 (.int (a + b)) :=
      Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup ha) (Cost.EvaluatesIn.lookup hb)
        (by simp [scalarBinary])
    have run := Cost.unpackPair (Cost.EvaluatesIn.lookup (module := program mode) hb) right
      "a" "b" (loop mode) []
    cases mode <;> exact run

private theorem iteration_cost (mode : Bool) (n a b t : Nat) (env : ScalarEnv)
    (valid : Locals mode n a b t env) (active : term mode a b ≤ n) :
    ∃ cost t' env', cost ≤ 37 ∧ Locals mode n b (a + b) t' env' ∧
      ReachesIn (step (program mode)) ⟨.block (loop mode) env, []⟩ cost
        ⟨.block (loop mode) env', []⟩ := by
  have guard := guard_continue mode n a b t env valid active
  have getTerm := term_lookup mode n a b t env valid
  have parity : Cost.EvaluatesIn (program mode) env
      (.compare (.binOp (.name (termName mode)) .mod (.constant (.int 2)))
        [(.eq, .constant (.int 0))]) 9 (.bool (decide (term mode a b % 2 = 0))) := by
    apply Cost.EvaluatesIn.compare (ca := 5) (cb := 1) (av := .int ↑(term mode a b % 2)) (bv := .int 0)
    · exact Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup getTerm) Cost.EvaluatesIn.constant
        (by simp [scalarBinary])
    · exact Cost.EvaluatesIn.constant
    · simp [scalarCompare, evalCmpOp]
      omega
  have branch := Cost.branch parity [.augAssign (.name "total") .add (.name (termName mode))]
    [] (afterParity mode) []
  by_cases even : term mode a b % 2 = 0
  · have add : Cost.EvaluatesIn (program mode) env
        (.binOp (.name "total") .add (.name (termName mode))) 5 (.int (t + term mode a b)) :=
      Cost.EvaluatesIn.binary (Cost.EvaluatesIn.lookup valid.2.2.2) (Cost.EvaluatesIn.lookup getTerm)
        (by simp [scalarBinary])
    have assigned := Cost.augAssign add (afterParity mode) []
    have valid' : Locals mode n a b (t + term mode a b) (env.set "total" (.int (t + term mode a b))) := by
      obtain ⟨hn, ha, hb, ht⟩ := valid
      cases mode <;> simpa [Locals, ScalarEnv.set, limitName] using And.intro hn (And.intro ha hb)
    obtain ⟨env', valid', unpack⟩ := unpack_cost mode n a b (t + term mode a b) _ valid'
    have bodyRun : ReachesIn (step (program mode)) ⟨.block (afterGuard mode) env, []⟩ 27
        ⟨.block (loop mode) env', []⟩ := by
      have branch' := branch
      simp only [even, decide_true, ↓reduceIte] at branch'
      cases mode <;> exact (branch'.trans assigned).trans unpack
    exact ⟨_, _, env', by cases mode <;> decide, valid', guard.trans bodyRun⟩
  · obtain ⟨env', valid', unpack⟩ := unpack_cost mode n a b t env valid
    have bodyRun : ReachesIn (step (program mode)) ⟨.block (afterGuard mode) env, []⟩ 20
        ⟨.block (loop mode) env', []⟩ := by
      have branch' := branch
      simp only [even, decide_false, Bool.false_eq_true, ↓reduceIte, List.nil_append] at branch'
      cases mode <;> exact branch'.trans unpack
    exact ⟨_, _, env', by cases mode <;> decide, valid', guard.trans bodyRun⟩

/-- Two source iterations change `(a,b)` to `(a+b,a+2*b)`. Since `a<b`,
the first component at least doubles, consuming one power-of-two budget. -/
private theorem loop_bound (mode : Bool) (n k a b t : Nat) (env : ScalarEnv)
    (positive : 0 < a) (ordered : a < b) (budget : n < 2 ^ k * a)
    (valid : Locals mode n a b t env) :
    ∃ cost value, FinishesIn (step (program mode)) ⟨.block (loop mode) env, []⟩ cost (.ok value) ∧
      cost ≤ 74 * k + 13 := by
  have exitBound : (if mode then 10 else 13) ≤ 13 := by cases mode <;> decide
  induction k generalizing a b t env with
  | zero =>
      have finished : n < term mode a b := by
        simp only [Nat.pow_zero, Nat.one_mul] at budget
        cases mode <;> simp only [term, Bool.false_eq_true, ↓reduceIte] <;> omega
      exact ⟨_, .int t, guard_exit mode n a b t env valid finished, by cases mode <;> decide⟩
  | succ k ih =>
      by_cases active : term mode a b ≤ n
      · obtain ⟨firstCost, t', env', firstBound, valid', first⟩ :=
          iteration_cost mode n a b t env valid active
        by_cases active' : term mode b (a + b) ≤ n
        · obtain ⟨secondCost, t'', env'', secondBound, valid'', second⟩ :=
            iteration_cost mode n b (a + b) t' env' valid' active'
          have nextBudget : n < 2 ^ k * (a + b) := by
            rw [Nat.pow_succ, Nat.mul_assoc] at budget
            exact Nat.lt_of_lt_of_le budget (Nat.mul_le_mul_left _ (by omega : 2 * a ≤ a + b))
          obtain ⟨cost, value, tail, bounded⟩ := ih (a + b) (b + (a + b)) t'' env''
            (by omega) (by omega) nextBudget valid''
          exact ⟨_, value, (first.trans second).finishes tail, by omega⟩
        · have tail := guard_exit mode n b (a + b) t' env' valid' (by omega)
          exact ⟨_, .int t', first.finishes tail, by omega⟩
      · have tail := guard_exit mode n a b t env valid (by omega)
        exact ⟨_, .int t, tail, by omega⟩

private def initialEnv (mode : Bool) (n : Nat) : ScalarEnv :=
  let args := argumentEnv [⟨limitName mode, some (.name "int"), none⟩] [.int n]
  if mode then ((args.set "a" (.int 1)).set "b" (.int 2)).set "total" (.int 0)
  else ((args.set "total" (.int 0)).set "a" (.int 1)).set "b" (.int 2)

private theorem initial_cost (mode : Bool) (n : Nat) :
    ReachesIn (step (program mode)) ⟨.invoke (functionName mode) [.int n], []⟩
      (if mode then 9 else 10) ⟨.block (loop mode) (initialEnv mode n), []⟩ := by
  cases mode <;> repeat first | exact ReachesIn.refl | refine ReachesIn.next rfl ?_

/-- Successful source execution, with a bound valid even at numeric bound zero. -/
theorem source_bound (mode : Bool) (n : Nat) :
    ∃ cost value, FinishesIn (step (program mode))
      ⟨.invoke (functionName mode) [.int n], []⟩ cost (.ok value) ∧
      cost ≤ 74 * (Nat.log2 n + 1) + 23 := by
  have budget : n < 2 ^ (Nat.log2 n + 1) * 1 := by
    rw [Nat.mul_one]
    by_cases zero : n = 0
    · subst n; decide
    · exact (Nat.log2_lt zero).mp (by omega)
  have valid : Locals mode n 1 2 0 (initialEnv mode n) := by
    cases mode <;> exact ⟨rfl, rfl, rfl, rfl⟩
  obtain ⟨cost, value, run, bounded⟩ := loop_bound mode n (Nat.log2 n + 1) 1 2 0
    (initialEnv mode n) (by decide) (by decide) budget valid
  have setupBound : (if mode then 9 else 10) ≤ 10 := by cases mode <;> decide
  exact ⟨_, value, (initial_cost mode n).finishes run, by omega⟩

def optimized (n : Nat) : Execution Constant := function Solvers.P2Ast.module "solve" [.int n]
def reference (n : Nat) : Execution Constant := function P2Ast.module "naive" [.int n]

/-- O(log n) in the numeric bound on the optimized source's declared domain. -/
theorem optimized_bigO : Execution.IsBigO optimized (fun n => 1 < n) id Nat.log2 := by
  refine ⟨171, 2, fun n _ large => ?_⟩
  obtain ⟨cost, value, run, bounded⟩ := source_bound true n
  have logPositive : 1 ≤ Nat.log2 n := (Nat.le_log2 (by omega)).mpr (by simpa using large)
  exact ⟨cost, .ok value, run, by simp only [id_eq]; omega⟩

/-- The reference also has logarithmic cost; its extra guard is constant per iteration. -/
theorem reference_bigO : Execution.IsBigO reference (fun _ => True) id Nat.log2 := by
  refine ⟨171, 2, fun n _ large => ?_⟩
  simp only [id_eq] at large
  obtain ⟨cost, value, run, bounded⟩ := source_bound false n
  have logPositive : 1 ≤ Nat.log2 n := (Nat.le_log2 (by omega)).mpr (by simpa using large)
  exact ⟨cost, .ok value, run, by simp only [id_eq]; omega⟩

end solvers.«2».complexity
