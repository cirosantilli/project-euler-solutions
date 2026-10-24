import «.formalsw».solvers.«61.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p61.py»
import FormalSW

namespace solvers.«61».polygonal
open FormalSW.PyAst

private abbrev answer :=
  data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.answer

/-- The actual helper preserves any surrounding loop's heap and global context. -/
theorem heap_result (s n : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    HeapCalls.ReturnsValue Solvers.P61Ast.module "polygonal" [.scalar (.int s), .scalar (.int n)] heap
      (.scalar (.int (answer s n))) heap globals := by
  let params := ((findFunctionIn Solvers.P61Ast.module.body "polygonal").getD ([], [])).1
  let body := ((findFunctionIn Solvers.P61Ast.module.body "polygonal").getD ([], [])).2
  let env := HeapCalls.functionEnv params body [.scalar (.int s), .scalar (.int n)] globals
  apply HeapCalls.BlockReturns.function rfl rfl
  change HeapCalls.BlockReturns Solvers.P61Ast.module env
    [.return (data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.expression "s" "n")]
    heap (.scalar (.int (answer s n))) heap globals
  apply HeapCalls.BlockReturns.return_value
  exact data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.expression_evaluates
    _ _ _ _ _ _ _ _ (by cases globals <;> rfl) (by cases globals <;> rfl)

/-- Source-level arithmetic, including Python's signed floor division. -/
theorem result (s n : Int) :
    Finishes (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).step
      (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).initial
      (.ok (.int (answer s n))) := by
  apply Execution.finishes_of_observe (fuel := 1)
  rfl

theorem «Returns» (s n : Int) :
    (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).Returns :=
  ⟨.int (answer s n), result s n⟩

theorem «Equiv(reference_polygonal)» (s n : Int) :
    (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).Equiv
      (scalarFunction P61Ast.module "polygonal" [.int s, .int n]) := by
  have reference :=
    data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.result s n
  have optimized := result s n
  intro outcome
  constructor
  · intro finished
    rw [finished.unique optimized]
    exact reference
  · intro finished
    rw [finished.unique reference]
    exact optimized

theorem «Supported» :
    (∀ s n : Int, (scalarFunction Solvers.P61Ast.module "polygonal" [.int s, .int n]).Supported) ∧
      (∀ s n : Int, (scalarFunction P61Ast.module "polygonal" [.int s, .int n]).Supported) :=
  ⟨fun s n => («Returns» s n).supported,
    data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.«Supported»⟩

end solvers.«61».polygonal

namespace solvers.«61».generate_4digit_polygonals
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)
private abbrev polygonal :=
  data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.answer

def accepted (value : Int) : Prop := 1000 ≤ value ∧ value ≤ 9999 ∧ 10 ≤ value % 100

instance (value : Int) : Decidable (accepted value) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

def emitted (value : Int) : List Int := if accepted value then [value] else []

/-- The bound follows from polygonal growth, rather than restricting the source input. -/
def collect : Nat → Int → Int → List Int
  | 0, _, _ => []
  | fuel + 1, sides, index =>
      let value := polygonal sides index
      if 10000 ≤ value then [] else emitted value ++ collect fuel sides (index + 1)

def answer (sides : Int) : List Int := collect 140 sides 1

private def numbers (values : List Int) : List Mutable.Value := values.map number
private def working (base : Mutable.Heap) (values : List Int) : Mutable.Heap :=
  base ++ [.list (numbers values)]

private def rangeCondition : Expr :=
  .compare (.constant (.int 1000)) [(.ltE, .name "val"), (.ltE, .constant (.int 9999))]
private def suffixCondition : Expr :=
  .compare (.binOp (.name "val") .mod (.constant (.int 100))) [(.gtE, .constant (.int 10))]
private def filterCode : List Stmt :=
  [.ifThen rangeCondition
    [.ifThen suffixCondition [.expr (.call (.attribute (.name "nums") "append") [.name "val"])] []] []]
private def body : List Stmt :=
  [.assign [.name "val"] (.call (.name "polygonal") [.name "s", .name "n"]),
   .ifThen (.compare (.name "val") [(.gtE, .constant (.int 10000))]) [.breakLoop] []] ++
  filterCode ++ [.augAssign (.name "n") .add (.constant (.int 1))]
private def statement : Stmt := .whileDo (.constant (.bool true)) body []
private def code : List Stmt :=
  [.annAssign (.name "nums") (.subscript (.name "list") (.name "int")) (some (.list [])),
   .assign [.name "n"] (.constant (.int 1)), statement, .return (.name "nums")]

theorem code_source :
    executableBody ((findFunctionIn program.body "generate_4digit_polygonals").getD ([], [])).2 = code := rfl

private theorem range_evaluates (value : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getVal : env "val" = some (number value)) (globals : Mutable.Globals) :
    Evaluates program env rangeCondition heap (flag (decide (1000 ≤ value ∧ value ≤ 9999))) heap globals := by
  by_cases lower : 1000 ≤ value
  · simp only [lower, true_and]
    apply Evaluates.chain (middle := heap) Evaluates.constant
    apply Evaluates.Compares.next (middle := heap) (afterComparison := heap) (compared := flag true)
      (Evaluates.lookup getVal)
    · simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp, lower]
    · rfl
    · apply Evaluates.Compares.last (middle := heap) (finalHeap := heap) Evaluates.constant
      simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]
  · simp only [lower, false_and, decide_false]
    apply Evaluates.chain (middle := heap) Evaluates.constant
    apply Evaluates.Compares.stop (middle := heap) (finalHeap := heap) (Evaluates.lookup getVal)
    · simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp, lower]
    · rfl

private theorem suffix_evaluates (value : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getVal : env "val" = some (number value)) (globals : Mutable.Globals) :
    Evaluates program env suffixCondition heap (flag (decide (10 ≤ value % 100))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (finalHeap := heap)
    (av := number (value % 100)) (bv := number 10)
  · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup getVal) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, flag, scalarCompare, evalCmpOp]

/-- The source's two filters append exactly one qualifying value, without allocating. -/
private theorem filter_step (value : Int) (env : HeapCalls.Env) (base : Mutable.Heap)
    (before : List Int) (getVal : env "val" = some (number value))
    (getNums : env "nums" = some (.ref base.length)) (globals : Mutable.Globals) :
    BlockContinues program env filterCode (working base before) env
      (working base (before ++ emitted value)) globals := by
  have rangeRun := range_evaluates value env (working base before) getVal globals
  have suffixRun := suffix_evaluates value env (working base before) getVal globals
  by_cases range : 1000 ≤ value ∧ value ≤ 9999
  · by_cases suffix : 10 ≤ value % 100
    · simp only [emitted, accepted, range.1, range.2, suffix, and_self, ↓reduceIte]
      apply BlockContinues.branch (flag := true) (by simpa [range] using rangeRun) rfl
      apply BlockContinues.branch (flag := true) (by simpa [suffix] using suffixRun) rfl
      have found : (working base before)[base.length]? = some (.list (numbers before)) := by simp [working]
      simpa [working, numbers, List.set_append_right] using
        BlockContinues.discard (Evaluates.append (Evaluates.lookup getNums) found (Evaluates.lookup getVal) found)
    · have rejected : ¬ accepted value := by simp [accepted, suffix]
      simp only [emitted, if_neg rejected, List.append_nil]
      apply BlockContinues.branch (flag := true) (by simpa [range] using rangeRun) rfl
      apply BlockContinues.branch (flag := false) (by simpa [suffix] using suffixRun) rfl
      exact .nil
  · have rejected : ¬ accepted value := fun h => range ⟨h.1, h.2.1⟩
    simp only [emitted, if_neg rejected, List.append_nil]
    apply BlockContinues.branch (flag := false) (by simpa [range] using rangeRun) rfl
    exact .nil

private theorem polygonal_call (s n : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getS : env "s" = some (number s)) (getN : env "n" = some (number n))
    (globals : Mutable.Globals) :
    Evaluates program env (.call (.name "polygonal") [.name "s", .name "n"]) heap
      (number (polygonal s n)) heap globals :=
  Evaluates.call (.cons (Evaluates.lookup getS) (.cons (Evaluates.lookup getN) .nil))
    (solvers.«61».polygonal.heap_result s n heap globals)

private theorem loop_done (s n : Int) (done : 10000 ≤ polygonal s n)
    (env : HeapCalls.Env) (base : Mutable.Heap) (before : List Int)
    (getS : env "s" = some (number s)) (getN : env "n" = some (number n))
    (getNums : env "nums" = some (.ref base.length)) (globals : Mutable.Globals) :
    BlockReturns program env [statement, .return (.name "nums")] (working base before)
      (.ref base.length) (working base before) globals := by
  apply BlockReturns.while_loop (flag := true) (middle := working base before) Evaluates.constant rfl
  apply BlockReturns.assign (polygonal_call s n env (working base before) getS getN globals)
  apply BlockReturns.branch (flag := true) (value := flag true) (middle := working base before)
    (Evaluates.compare (middle := working base before) (afterRight := working base before)
      (av := number (polygonal s n)) (bv := number 10000)
      (Evaluates.lookup rfl) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag, done])) rfl
  apply BlockReturns.break_loop (next := statement) (following := [.return (.name "nums")]) rfl
  exact BlockReturns.return_value (Evaluates.lookup (by simpa [setLocal] using getNums))

/-- Every actual loop call reaches its break by index 141 and preserves the caller heap prefix. -/
theorem loop (fuel : Nat) (s n : Int) (sides : 3 ≤ s) (positive : 1 ≤ n)
    (bound : 141 ≤ n + fuel) (env : HeapCalls.Env) (base : Mutable.Heap) (before : List Int)
    (getS : env "s" = some (number s)) (getN : env "n" = some (number n))
    (getNums : env "nums" = some (.ref base.length)) (globals : Mutable.Globals := []) :
    BlockReturns program env [statement, .return (.name "nums")] (working base before)
      (.ref base.length) (working base (before ++ collect fuel s n)) globals := by
  induction fuel generalizing n env before with
  | zero =>
      have done := data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.above_four_digits
        s n sides (by simpa using bound)
      simpa [collect] using loop_done s n done env base before getS getN getNums globals
  | succ fuel ih =>
      by_cases done : 10000 ≤ polygonal s n
      · simpa [collect, done] using loop_done s n done env base before getS getN getNums globals
      · let valued := setLocal env "val" (number (polygonal s n))
        let nextEnv := setLocal valued "n" (number (n + 1))
        have valuedS : valued "s" = some (number s) := by simpa [valued, setLocal] using getS
        have valuedN : valued "n" = some (number n) := by simpa [valued, setLocal] using getN
        have valuedNums : valued "nums" = some (.ref base.length) := by simpa [valued, setLocal] using getNums
        have nextS : nextEnv "s" = some (number s) := by simpa [nextEnv, setLocal] using valuedS
        have nextN : nextEnv "n" = some (number (n + 1)) := by simp [nextEnv, setLocal]
        have nextNums : nextEnv "nums" = some (.ref base.length) := by simpa [nextEnv, setLocal] using valuedNums
        have filtered := filter_step (polygonal s n) valued base before rfl valuedNums globals
        have rest := ih (n + 1) (by omega) (by omega) nextEnv
          (before ++ emitted (polygonal s n)) nextS nextN nextNums
        simp only [collect, if_neg done]
        apply BlockReturns.while_loop (flag := true) (middle := working base before) Evaluates.constant rfl
        apply BlockReturns.assign (polygonal_call s n env (working base before) getS getN globals)
        apply BlockReturns.branch (flag := false) (value := flag false) (middle := working base before)
          (Evaluates.compare (middle := working base before) (afterRight := working base before)
            (av := number (polygonal s n)) (bv := number 10000)
            (Evaluates.lookup rfl) Evaluates.constant
            (by simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag, done])) rfl
        apply filtered.returns
        apply BlockReturns.augAssign valuedN Evaluates.constant
          (value := number (n + 1)) (afterOperation := working base (before ++ emitted (polygonal s n)))
        · simp [HeapCalls.inplace, Mutable.binary, scalarBinary, number]
        · apply BlockReturns.loop_resume
          simpa [nextEnv, List.append_assoc] using rest

/-- The unchanged Python helper returns its exact list for every valid polygonal family. -/
theorem result (s : Int) (sides : 3 ≤ s) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    ReturnsValue program "generate_4digit_polygonals" [number s] heap (.ref heap.length)
      (working heap (answer s)) globals := by
  let params := ((findFunctionIn program.body "generate_4digit_polygonals").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "generate_4digit_polygonals").getD ([], [])).2
  let env := functionEnv params sourceBody [number s] globals
  apply BlockReturns.function rfl rfl
  change BlockReturns program env (executableBody sourceBody) heap (.ref heap.length) (working heap (answer s)) globals
  rw [code_source]
  have allocated : BlockContinues program env
      [.annAssign (.name "nums") (.subscript (.name "list") (.name "int")) (some (.list []))]
      heap (setLocal env "nums" (.ref heap.length)) (working heap []) globals := by
    exact BlockContinues.annAssign (Evaluates.list .nil)
  apply allocated.returns
  apply BlockReturns.assign Evaluates.constant
  exact loop 140 s 1 sides (by decide) (by decide)
    (setLocal (setLocal env "nums" (.ref heap.length)) "n" (number 1)) heap []
    (by cases globals <;> rfl) rfl rfl globals

end solvers.«61».generate_4digit_polygonals
