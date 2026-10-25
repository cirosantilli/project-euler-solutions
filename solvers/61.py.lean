import «.formalsw».solvers.«61.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p61.py»
import FormalSW
import ProjectEulerSolutions.Lib.PolygonalCycle
import ProjectEulerSolutions.Lib.PolygonalCycleOptimized

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

/-- Later positive indices never decrease within a polygonal family. -/
theorem monotone_from (s n : Int) (sides : 3 ≤ s) (positive : 1 ≤ n) (offset : Nat) :
    polygonal s n ≤ polygonal s (n + offset) := by
  induction offset with
  | zero => simp
  | succ offset ih =>
      have step := data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.increasing_step
        s (n + offset) sides (by omega)
      exact ih.trans (by simpa [Nat.cast_add, Nat.cast_one, add_assoc] using step.le)

/-- Early termination discards only values which fail the four-digit filter. -/
theorem collect_eq_window (fuel : Nat) (s n : Int) (sides : 3 ≤ s) (positive : 1 ≤ n) :
    collect fuel s n =
      ((List.range fuel).map (fun offset : Nat => polygonal s (n + offset))).filter
        (fun value => decide (accepted value)) := by
  induction fuel generalizing n with
  | zero => rfl
  | succ fuel ih =>
      have mapped : (List.range (fuel + 1)).map (fun offset : Nat => polygonal s (n + offset)) =
          polygonal s n :: (List.range fuel).map (fun offset : Nat => polygonal s (n + 1 + offset)) := by
        rw [List.range_succ_eq_map, List.map_cons, List.map_map]
        simp only [Nat.cast_zero, add_zero]
        congr 1
        apply List.map_congr_left
        intro offset _
        dsimp
        congr 1
        ring
      rw [mapped]
      by_cases done : 10000 ≤ polygonal s n
      · have rejected (offset : Nat) : ¬ accepted (polygonal s (n + offset)) := by
          have above := done.trans (monotone_from s n sides positive offset)
          intro accepted
          have upper := accepted.2.1
          omega
        have first : ¬ accepted (polygonal s n) := by simpa using rejected 0
        have tail :
            ((List.range fuel).map (fun offset : Nat => polygonal s (n + 1 + offset))).filter
              (fun value => decide (accepted value)) = [] := by
          apply List.filter_eq_nil_iff.mpr
          intro value member
          obtain ⟨offset, _, rfl⟩ := List.mem_map.mp member
          have absent : ¬ accepted (polygonal s (n + 1 + offset)) := by
            simpa [Nat.cast_add, Nat.cast_one, add_assoc, add_comm, add_left_comm] using rejected (offset + 1)
          simpa using absent
        simp [collect, done, first, tail]
      · rw [collect, if_neg done, ih (n + 1) (by omega)]
        by_cases keep : accepted (polygonal s n) <;> simp [emitted, keep]

/-- The generated list contains every qualifying positive-index polygonal number. -/
theorem mem_answer (s value : Int) (sides : 3 ≤ s) :
    value ∈ answer s ↔
      ∃ n : Nat, 1 ≤ n ∧ n ≤ 140 ∧ polygonal s n = value ∧ accepted value := by
  rw [answer, collect_eq_window 140 s 1 sides (by decide)]
  simp only [List.mem_filter, decide_eq_true_eq, List.mem_map, List.mem_range]
  constructor
  · rintro ⟨⟨offset, bound, equal⟩, keep⟩
    refine ⟨offset + 1, by omega, by omega, ?_, keep⟩
    simpa [Nat.cast_add, Nat.cast_one, add_comm] using equal
  · rintro ⟨n, positive, upper, equal, keep⟩
    refine ⟨⟨n - 1, by omega, ?_⟩, keep⟩
    have index : (1 : Int) + ↑(n - 1) = n := by omega
    simpa [index] using equal

/-- No positive index outside the finite window can add a qualifying value. -/
theorem mem_answer_iff (s value : Int) (sides : 3 ≤ s) :
    value ∈ answer s ↔
      ∃ n : Nat, 1 ≤ n ∧ polygonal s n = value ∧ accepted value := by
  rw [mem_answer s value sides]
  constructor
  · rintro ⟨n, positive, _, equal, keep⟩
    exact ⟨n, positive, equal, keep⟩
  · rintro ⟨n, positive, equal, keep⟩
    refine ⟨n, positive, ?_, equal, keep⟩
    by_contra outside
    have above := data.«project-euler-statements».data.py.project_euler_statements.p61.polygonal.above_four_digits
      s n sides (by omega)
    have upper := keep.2.1
    change 10000 ≤ polygonal s n at above
    rw [equal] at above
    omega

theorem candidates_eq_model (s : Int) (sides : 3 ≤ s) :
    answer s = ProjectEulerSolutions.PolygonalCycle.candidates s := by
  rw [answer, collect_eq_window 140 s 1 sides (by decide)]
  rfl

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

namespace solvers.«61»

theorem generate_4digit_polygonals_source_result (s : Int) (heap : FormalSW.PyAst.Mutable.Heap)
    (globals : FormalSW.PyAst.Mutable.Globals) :
    3 ≤ s → FormalSW.PyAst.HeapCalls.ReturnsValue Solvers.P61Ast.module "generate_4digit_polygonals"
      [.scalar (.int s)] heap (.ref heap.length)
      (heap ++ [.list ((solvers.«61».generate_4digit_polygonals.answer s).map
        (fun value => .scalar (.int value)))]) globals := by
  intro sides
  exact solvers.«61».generate_4digit_polygonals.result s sides heap globals

end solvers.«61»

namespace solvers.«61».dfs
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

/-- Only the two endpoint tuples are read by a six-node DFS call. -/
structure Ends (heap : Mutable.Heap) (path : Nat) (first last : Int) where
  items : List Mutable.Value
  size : items.length = 6
  pathFound : heap[path]? = some (.list items)
  firstAddress : Nat
  lastAddress : Nat
  firstKind : Mutable.Value
  lastKind : Mutable.Value
  head : items[0]? = some (.ref firstAddress)
  tail : items[5]? = some (.ref lastAddress)
  firstFound : heap[firstAddress]? = some (.tuple [number first, firstKind])
  lastFound : heap[lastAddress]? = some (.tuple [number last, lastKind])

variable {heap : Mutable.Heap} {path : Nat} {first last : Int}

private def lastExpression : Expr :=
  .subscript (.subscript (.name "path") (.unaryOp .uSub (.constant (.int 1)))) (.constant (.int 0))
private def firstExpression : Expr :=
  .subscript (.subscript (.name "path") (.constant (.int 0))) (.constant (.int 0))
private def condition : Expr :=
  .compare (.builtinCall .len [.name "path"]) [(.eq, .constant (.int 6))]
private def closureCondition : Expr :=
  .compare (.binOp (.name "last_num") .mod (.constant (.int 100)))
    [(.eq, .binOp (.name "first_num") .floorDiv (.constant (.int 100)))]
private def closedBody : List Stmt :=
  [.assign [.name "last_num"] lastExpression,
   .assign [.name "first_num"] firstExpression,
   .ifThen closureCondition [.return (.name "path")] [],
   .return (.constant .none)]

theorem base_source :
    (executableBody ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2)[0]? =
      some (.ifThen condition closedBody []) := rfl

private theorem length_evaluates (ends : Ends heap path first last) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref path)) (globals : Mutable.Globals) :
    Evaluates program env condition heap (flag true) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (finalHeap := heap)
    (av := number 6) (bv := number 6)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getPath) .nil)
    simp [primitive, Mutable.length, ends.pathFound, ends.size, number]
  · exact Evaluates.constant
  · rfl

private theorem last_evaluates (ends : Ends heap path first last) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref path)) (globals : Mutable.Globals) :
    Evaluates program env lastExpression heap (number last) heap globals := by
  have nonempty : ends.items ≠ [] := by intro empty; have size := ends.size; simp [empty] at size
  have inner : Evaluates program env
      (.subscript (.name "path") (.unaryOp .uSub (.constant (.int 1)))) heap (.ref ends.lastAddress) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getPath)
      (Evaluates.unary (middle := heap) (finalHeap := heap) (value := number 1) Evaluates.constant rfl)
    simp [Mutable.getItem, ends.pathFound, Mutable.index, ends.size, nonempty, Except.map]
    exact (List.getElem?_eq_some_iff.mp ends.tail).2
  apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
    (by intros; intro impossible; cases impossible) inner Evaluates.constant
  simp [Mutable.getItem, ends.lastFound, Mutable.index, Except.map]

private theorem first_evaluates (ends : Ends heap path first last) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref path)) (globals : Mutable.Globals) :
    Evaluates program env firstExpression heap (number first) heap globals := by
  have nonempty : ends.items ≠ [] := by intro empty; have size := ends.size; simp [empty] at size
  have inner : Evaluates program env (.subscript (.name "path") (.constant (.int 0))) heap
      (.ref ends.firstAddress) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getPath) Evaluates.constant
    simp [Mutable.getItem, ends.pathFound, Mutable.index, nonempty, ends.head, Except.map]
  apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
    (by intros; intro impossible; cases impossible) inner Evaluates.constant
  simp [Mutable.getItem, ends.firstFound, Mutable.index, Except.map]

private theorem closure_evaluates (first last : Int) (env : HeapCalls.Env) (heap : Mutable.Heap)
    (getFirst : env "first_num" = some (number first)) (getLast : env "last_num" = some (number last))
    (globals : Mutable.Globals) :
    Evaluates program env closureCondition heap (flag (decide (last % 100 = first / 100))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (finalHeap := heap)
    (av := number (last % 100)) (bv := number (first / 100))
  · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup getLast) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.lookup getFirst) Evaluates.constant
    simp [Mutable.binary, scalarBinary, number]
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]

/-- A completed path returns its existing list exactly when its endpoints link. -/
theorem closed_result (heap : Mutable.Heap) (path : Nat) (first last : Int) (ends : Ends heap path first last)
    (usedTypes usedNums prefixCell typesCell : Mutable.Value) (globals : Mutable.Globals := []) :
    ReturnsValue program "__formalsw_local_solve_dfs" [.ref path, usedTypes, usedNums, prefixCell, typesCell]
      heap (if last % 100 = first / 100 then .ref path else .scalar .none) heap globals := by
  let params := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2
  let env := functionEnv params sourceBody [.ref path, usedTypes, usedNums, prefixCell, typesCell] globals
  let valued := setLocal env "last_num" (number last)
  let bound := setLocal valued "first_num" (number first)
  have getPath : env "path" = some (.ref path) := by cases globals <;> rfl
  have valuedPath : valued "path" = some (.ref path) := by simpa [valued, setLocal] using getPath
  have boundPath : bound "path" = some (.ref path) := by simpa [bound, setLocal] using valuedPath
  have boundFirst : bound "first_num" = some (number first) := rfl
  have boundLast : bound "last_num" = some (number last) := by simp [bound, valued, setLocal]
  have closing := closure_evaluates first last bound heap boundFirst boundLast globals
  apply BlockReturns.function rfl rfl
  change BlockReturns program env
    (.ifThen condition closedBody [] :: (executableBody sourceBody).drop 1) heap _ heap globals
  apply BlockReturns.branch (flag := true) (value := flag true) (length_evaluates ends env getPath globals) rfl
  apply BlockReturns.assign (last_evaluates ends env getPath globals)
  apply BlockReturns.assign (first_evaluates ends valued valuedPath globals)
  by_cases closes : last % 100 = first / 100
  · simp only [if_pos closes]
    apply BlockReturns.branch (flag := true) (by simpa [closes] using closing) rfl
    exact BlockReturns.return_value (Evaluates.lookup boundPath)
  · simp only [if_neg closes]
    apply BlockReturns.branch (flag := false) (by simpa [closes] using closing) rfl
    exact BlockReturns.return_value Evaluates.constant

end solvers.«61».dfs

namespace solvers.«61».solve_setup
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private def numbers (values : List Int) : List Mutable.Value := values.map number
private abbrev candidates := solvers.«61».generate_4digit_polygonals.answer

def types : List Int := [3, 4, 5, 6, 7, 8]
def typeCell : Expr := .subscript (.name "__formalsw_cell_solve_types") (.constant (.int 0))
def numberExpression : Expr := .call (.name "generate_4digit_polygonals") [.name "t"]
def numberGenerators : List (Expr × Expr × List Expr) := [(.name "t", typeCell, [])]
def numberComprehension : Expr := .dictComp (.name "t") numberExpression numberGenerators

private def working (base : Mutable.Heap) (entries : List (Constant × Mutable.Value)) (buffers : Mutable.Heap) :
    Mutable.Heap := base ++ (.dict entries :: buffers)

private def update (baseSize : Nat) (entries : List (Constant × Mutable.Value)) (buffers : Mutable.Heap)
    (sides : Int) : List (Constant × Mutable.Value) × Mutable.Heap :=
  (ScalarDict.insert entries (.int sides) (.ref (baseSize + 1 + buffers.length)),
   buffers ++ [.list (numbers (candidates sides))])

private def generateModel (baseSize : Nat) : List Int → List (Constant × Mutable.Value) → Mutable.Heap →
    List (Constant × Mutable.Value) × Mutable.Heap
  | [], entries, buffers => (entries, buffers)
  | sides :: rest, entries, buffers =>
      let next := update baseSize entries buffers sides
      generateModel baseSize rest next.1 next.2

private def numberBody (address : Nat) : List Stmt :=
  [.comprehensionStore address (.name "t") numberExpression]

private theorem number_store (base : Mutable.Heap) (entries : List (Constant × Mutable.Value))
    (buffers : Mutable.Heap) (sides : Int) (valid : 3 ≤ sides) (env : HeapCalls.Env)
    (getType : env "t" = some (number sides)) (globals : Mutable.Globals) :
    BlockContinues program env (numberBody base.length) (working base entries buffers) env
      (working base (update base.length entries buffers sides).1 (update base.length entries buffers sides).2)
      globals := by
  let heap := working base entries buffers
  have evaluated : Evaluates program env numberExpression heap (.ref heap.length)
      (heap ++ [.list (numbers (candidates sides))]) globals :=
    Evaluates.call (.cons (Evaluates.lookup getType) .nil)
      (solvers.«61».generate_4digit_polygonals.result sides valid heap globals)
  have table : (heap ++ [.list (numbers (candidates sides))])[base.length]? = some (.dict entries) := by
    simp [heap, working, List.append_assoc]
  apply BlockContinues.comprehension_store (.cons (Evaluates.lookup getType) (.cons evaluated .nil))
  have written := Mutable.setItem_dictionary _ _ _ table (.int sides) (.ref heap.length)
  simpa [update, heap, working, number, List.set_append_right, List.length_append, List.append_assoc,
    Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using written

private theorem number_loop (remaining values : List Int) (position : Nat)
    (suffix : values.drop position = remaining) (valid : ∀ s ∈ remaining, (3 : Int) ≤ s)
    (base : Mutable.Heap) (input : Nat) (found : base[input]? = some (.list (numbers values)))
    (entries : List (Constant × Mutable.Value)) (buffers : Mutable.Heap) (env : HeapCalls.Env)
    (globals : Mutable.Globals) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "t" (.ref input) position (numberBody base.length) []]
      (working base entries buffers) finalEnv
      (working base (generateModel base.length remaining entries buffers).1
        (generateModel base.length remaining entries buffers).2) globals := by
  have inside := (List.getElem?_eq_some_iff.mp found).1
  have sequence (entries : List (Constant × Mutable.Value)) (buffers : Mutable.Heap) :
      Mutable.sequence (.ref input) (working base entries buffers) =
        some (.ok (numbers values), working base entries buffers) := by
    simp [working, Mutable.sequence, List.getElem?_append_left inside, found]
  induction remaining generalizing position entries buffers env with
  | nil =>
      refine ⟨env, ?_⟩
      apply BlockContinues.sequence_done (items := numbers values)
        (by simpa only [sequenceValue] using sequence entries buffers)
      · have done : values[position]? = none := by rw [← List.head?_drop, suffix]; rfl
        simp [numbers, List.getElem?_map, done]
      · exact .nil
  | cons sides rest ih =>
      let bound := setLocal env "t" (number sides)
      have generated := number_store base entries buffers sides (valid sides (by simp)) bound rfl globals
      obtain ⟨finalEnv, executed⟩ := ih (position + 1)
        (by simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons])
        (by intro s member; exact valid s (by simp [member]))
        (update base.length entries buffers sides).1 (update base.length entries buffers sides).2 bound
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.sequence_step (items := numbers values) (value := number sides)
        (by simpa only [sequenceValue] using sequence entries buffers)
      · have item : values[position]? = some sides := by rw [← List.head?_drop, suffix]; rfl
        simp [numbers, List.getElem?_map, item]
      · simpa [numberBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, generateModel]
          using generated.append executed

/-- Every comprehension callback runs the actual polygonal-list helper in the shared heap. -/
theorem numbers_evaluate (values : List Int) (valid : ∀ s ∈ values, (3 : Int) ≤ s)
    (heap : Mutable.Heap) (input cell : Nat) (env : HeapCalls.Env)
    (getCell : env "__formalsw_cell_solve_types" = some (.ref cell))
    (cellFound : heap[cell]? = some (.list [.ref input]))
    (found : heap[input]? = some (.list (numbers values))) (globals : Mutable.Globals := []) :
    Evaluates program env numberComprehension heap (.ref heap.length)
      (working heap (generateModel heap.length values [] []).1 (generateModel heap.length values [] []).2)
      globals := by
  let masked := comprehensionTargetsEnv env numberGenerators
  obtain ⟨finalEnv, executed⟩ := number_loop values values 0 rfl valid heap input found [] [] masked globals
  have iterable : Evaluates program env typeCell heap (.ref input) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getCell) Evaluates.constant
    exact Mutable.getItem_list heap cell 0 [.ref input] cellFound (by simp)
  apply Evaluates.dict_comp_sequence (receiver := .ref input) (items := numbers values)
    (middle := heap) (finalEnv := finalEnv)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    iterable (by simp [Mutable.sequence, found])
  simpa only [numberGenerators, comprehensionTargetName, comprehensionTargetBody, comprehensionFilters,
    List.foldr_nil, comprehensionTargetsBody, sequenceSource, masked, numberBody, working] using executed

def initialHeap (caller : Mutable.Heap) : Mutable.Heap :=
  caller ++ [.list (numbers types), .list [.ref caller.length]]

def initialEnv (env : HeapCalls.Env) (caller : Mutable.Heap) : HeapCalls.Env :=
  setLocal env "__formalsw_cell_solve_types" (.ref (caller.length + 1))

def initialCode : List Stmt :=
  [.assign [.name "__formalsw_cell_solve_types"]
    (.list [.list [.constant (.int 3), .constant (.int 4), .constant (.int 5),
      .constant (.int 6), .constant (.int 7), .constant (.int 8)]])]

theorem initial_source :
    (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).take 1 = initialCode := rfl

theorem initial_result (caller : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals := []) :
    BlockContinues program env initialCode caller (initialEnv env caller) (initialHeap caller) globals := by
  apply BlockContinues.assign
  convert Evaluates.list (module := program) (env := env) (heap := caller) (globals := globals)
    (.cons (Evaluates.list (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant
      (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil))))))) .nil) using 1 <;>
    simp [initialHeap, types, numbers, List.append_assoc]

def generatedHeap (caller : Mutable.Heap) : Mutable.Heap :=
  let heap := initialHeap caller
  let generated := generateModel heap.length types [] []
  working heap generated.1 generated.2

def numberEntries (caller : Mutable.Heap) : List (Constant × Mutable.Value) :=
  types.map (fun sides => (.int sides, .ref (caller.length + sides.toNat)))

/-- The generated lists occupy six distinct objects after the type cell and dictionary. -/
theorem generated_layout (caller : Mutable.Heap) :
    generatedHeap caller = initialHeap caller ++
      (.dict (numberEntries caller) :: types.map (fun sides => .list (numbers (candidates sides)))) := by
  simp [generatedHeap, initialHeap, generateModel, update, working, types, numberEntries,
    ScalarDict.insert, scalarCompare, evalCmpOp, List.append_assoc, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

theorem generated_table (caller : Mutable.Heap) :
    (generatedHeap caller)[(initialHeap caller).length]? = some (.dict (numberEntries caller)) := by
  rw [generated_layout]
  simp

theorem generated_family (caller : Mutable.Heap) (sides : Int) (lower : 3 ≤ sides) (upper : sides ≤ 8) :
    (generatedHeap caller)[caller.length + sides.toNat]? = some (.list (numbers (candidates sides))) := by
  have cases : sides = 3 ∨ sides = 4 ∨ sides = 5 ∨ sides = 6 ∨ sides = 7 ∨ sides = 8 := by omega
  rcases cases with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [generated_layout, initialHeap, types, List.append_assoc]

theorem numberEntries_lookup (caller : Mutable.Heap) (sides : Int) (lower : 3 ≤ sides) (upper : sides ≤ 8) :
    ScalarDict.lookup (numberEntries caller) (.int sides) = some (.ref (caller.length + sides.toNat)) := by
  have cases : sides = 3 ∨ sides = 4 ∨ sides = 5 ∨ sides = 6 ∨ sides = 7 ∨ sides = 8 := by omega
  rcases cases with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [numberEntries, types, ScalarDict.lookup, scalarCompare, evalCmpOp]

def generatedEnv (env : HeapCalls.Env) (caller : Mutable.Heap) : HeapCalls.Env :=
  setLocal (initialEnv env caller) "nums_by_type" (.ref (initialHeap caller).length)

def generateCode : List Stmt :=
  [.annAssign (.name "nums_by_type")
    (.subscript (.name "dict") (.tuple [.name "int", .subscript (.name "list") (.name "int")]))
    (some numberComprehension)]

theorem generate_source :
    ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 1).take 1 = generateCode := rfl

/-- The first two actual solve statements construct all six candidate families and their shared type cell. -/
theorem generate_result (caller : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals := []) :
    BlockContinues program env (initialCode ++ generateCode) caller
      (generatedEnv env caller) (generatedHeap caller) globals := by
  apply (initial_result caller env globals).append
  apply BlockContinues.annAssign
  apply numbers_evaluate (globals := globals) types (by intro s member; simp [types] at member; omega)
    (initialHeap caller) caller.length (caller.length + 1) (initialEnv env caller) rfl
  · simp [initialHeap]
  · simp [initialHeap]

end solvers.«61».solve_setup

namespace solvers.«61»

theorem candidate_setup_source_result (caller : FormalSW.PyAst.Mutable.Heap)
    (env : FormalSW.PyAst.HeapCalls.Env) (globals : FormalSW.PyAst.Mutable.Globals) :
    FormalSW.PyAst.HeapCalls.BlockContinues Solvers.P61Ast.module env
      ((FormalSW.PyAst.executableBody
        ((FormalSW.PyAst.findFunctionIn Solvers.P61Ast.module.body "solve").getD ([], [])).2).take 2)
      caller (solve_setup.generatedEnv env caller) (solve_setup.generatedHeap caller) globals := by
  change FormalSW.PyAst.HeapCalls.BlockContinues Solvers.P61Ast.module env
    (solve_setup.initialCode ++ solve_setup.generateCode) caller
    (solve_setup.generatedEnv env caller) (solve_setup.generatedHeap caller) globals
  exact solve_setup.generate_result caller env globals

end solvers.«61»

namespace solvers.«61».build_prefix_map
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private def numbers (values : List Int) : List Mutable.Value := values.map number

def bucket (values : List Int) (key : Int) : List Int :=
  values.filter (fun value => decide (value / 100 = key))

def lookup : List (Int × Nat) → Int → Option Nat
  | [], _ => none
  | (stored, address) :: rest, key => if stored = key then some address else lookup rest key

private def encode (entries : List (Int × Nat)) : List (Constant × Mutable.Value) :=
  entries.map (fun (key, address) => (.int key, .ref address))

private theorem lookup_encode (entries : List (Int × Nat)) (key : Int) :
    ScalarDict.lookup (encode entries) (.int key) = (lookup entries key).map Mutable.Value.ref := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨stored, address⟩ := entry
      by_cases same : stored = key
      · simp [encode, lookup, ScalarDict.lookup, scalarCompare, evalCmpOp, same]
      · simpa [encode, lookup, ScalarDict.lookup, scalarCompare, evalCmpOp, same] using ih

private theorem lookup_append (left right : List (Int × Nat)) (key : Int) :
    lookup (left ++ right) key =
      match lookup left key with
      | some address => some address
      | none => lookup right key := by
  induction left with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨stored, address⟩ := entry
      by_cases same : stored = key <;> simp [lookup, same, ih]

private theorem insert_encode (entries : List (Int × Nat)) (key : Int) (address : Nat)
    (absent : lookup entries key = none) :
    ScalarDict.insert (encode entries) (.int key) (.ref address) = encode (entries ++ [(key, address)]) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨stored, previous⟩ := entry
      by_cases same : stored = key
      · simp [lookup, same] at absent
      · have tail : lookup rest key = none := by simpa [lookup, same] using absent
        simpa [encode, ScalarDict.insert, scalarCompare, evalCmpOp, same] using
          congrArg (fun result : List (Constant × Mutable.Value) => (.int stored, .ref previous) :: result) (ih tail)

private theorem found_append (heap extra : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1]
  exact found

private theorem extension_set (base heap : Mutable.Heap)
    (extension : ∃ extra, heap = base ++ extra) (address : Nat) (outside : base.length ≤ address)
    (object : Mutable.Object) : ∃ extra, heap.set address object = base ++ extra := by
  obtain ⟨extra, rfl⟩ := extension
  exact ⟨extra.set (address - base.length) object, List.set_append_right address object outside⟩

/-- The source table owns distinct fresh list objects. Its buckets retain input order. -/
structure Store (base heap : Mutable.Heap) (processed : List Int) where
  entries : List (Int × Nat)
  extension : ∃ extra, heap = base ++ extra
  table : heap[base.length]? = some (.dict (encode entries))
  present : ∀ key address, lookup entries key = some address →
    base.length < address ∧ heap[address]? = some (.list (numbers (bucket processed key)))
  exclusive : ∀ left right address,
    lookup entries left = some address → lookup entries right = some address → left = right
  missing : ∀ key, lookup entries key = none → bucket processed key = []

def Store.initial (base : Mutable.Heap) : Store base (base ++ [.dict []]) [] where
  entries := []
  extension := ⟨[.dict []], rfl⟩
  table := by simp [encode]
  present := by intro key address found; simp [lookup] at found
  exclusive := by intro left right address found; simp [lookup] at found
  missing := by intro key _; rfl

variable {base heap : Mutable.Heap} {processed : List Int}

def Store.append (store : Store base heap processed) (extra : Mutable.Heap) :
    Store base (heap ++ extra) processed where
  entries := store.entries
  extension := by
    obtain ⟨old, rfl⟩ := store.extension
    exact ⟨old ++ extra, List.append_assoc _ _ _⟩
  table := found_append heap extra _ _ store.table
  present := by
    intro key address looked
    obtain ⟨outside, found⟩ := store.present key address looked
    exact ⟨outside, found_append heap extra address _ found⟩
  exclusive := store.exclusive
  missing := store.missing

/-- Eager evaluation allocates a default even if another bucket already exists. -/
def Store.add_empty (store : Store base heap processed) (key : Int) (absent : lookup store.entries key = none) :
    Store base ((heap ++ [Mutable.Object.list []]).set base.length
      (.dict (encode (store.entries ++ [(key, heap.length)])))) processed := by
  let entries : List (Int × Nat) := store.entries ++ [(key, heap.length)]
  let extended : Mutable.Heap := heap ++ [.list []]
  let updated : Mutable.Heap := extended.set base.length (.dict (encode entries))
  have tableInside : base.length < heap.length := (List.getElem?_eq_some_iff.mp store.table).1
  have classify (query : Int) (address : Nat) (looked : lookup entries query = some address) :
      lookup store.entries query = some address ∨ (query = key ∧ address = heap.length) := by
    cases old : lookup store.entries query with
    | some previous =>
        have same : previous = address := by simpa [entries, lookup_append, old] using looked
        exact Or.inl (by simp [same])
    | none =>
        by_cases same : key = query
        · subst query
          have fresh : heap.length = address := by simpa [entries, lookup_append, old, lookup] using looked
          exact Or.inr ⟨rfl, fresh.symm⟩
        · simp [entries, lookup_append, old, lookup, same] at looked
  refine { entries := entries, extension := ?_, table := ?_, present := ?_, exclusive := ?_, missing := ?_ }
  · exact extension_set base extended (store.append [.list []]).extension _ (Nat.le_refl _) _
  · exact List.getElem?_set_self (by simp only [List.length_append, List.length_singleton]; omega)
  · intro query address looked
    rcases classify query address looked with old | ⟨rfl, rfl⟩
    · obtain ⟨outside, found⟩ := store.present query address old
      refine ⟨outside, ?_⟩
      simpa only [updated, List.getElem?_set_ne (by omega : base.length ≠ address)] using
        found_append heap [.list []] address _ found
    · refine ⟨tableInside, ?_⟩
      have empty := store.missing _ absent
      rw [List.getElem?_set_ne (by omega : base.length ≠ heap.length)]
      simp [empty, numbers]
  · intro left right address leftFound rightFound
    rcases classify left address leftFound with oldLeft | ⟨rfl, freshLeft⟩
    · rcases classify right address rightFound with oldRight | ⟨rfl, freshRight⟩
      · exact store.exclusive left right address oldLeft oldRight
      · have found := (store.present left address oldLeft).2
        have inside := (List.getElem?_eq_some_iff.mp found).1
        omega
    · rcases classify right address rightFound with oldRight | ⟨rfl, _⟩
      · have found := (store.present right address oldRight).2
        have inside := (List.getElem?_eq_some_iff.mp found).1
        omega
      · rfl
  · intro query missing
    have old : lookup store.entries query = none := by
      cases looked : lookup store.entries query with
      | none => rfl
      | some address => simp [entries, lookup_append, looked] at missing
    exact store.missing query old

/-- Mutating one bucket preserves the table, all other buckets, and the caller prefix. -/
def Store.bump (store : Store base heap processed) (value : Int) (address : Nat)
    (looked : lookup store.entries (value / 100) = some address) :
    Store base (heap.set address (.list (numbers (bucket processed (value / 100) ++ [value]))))
      (processed ++ [value]) := by
  obtain ⟨outside, found⟩ := store.present (value / 100) address looked
  refine { entries := store.entries
           extension := ?_
           table := ?_
           present := ?_
           exclusive := store.exclusive
           missing := ?_ }
  · exact extension_set base heap store.extension address (by omega) _
  · simpa only [List.getElem?_set_ne (by omega : address ≠ base.length)] using store.table
  · intro key other otherLooked
    obtain ⟨otherOutside, otherFound⟩ := store.present key other otherLooked
    refine ⟨otherOutside, ?_⟩
    by_cases same : key = value / 100
    · subst key
      have equal : other = address := Option.some.inj (otherLooked.symm.trans looked)
      subst other
      simpa [bucket] using List.getElem?_set_self (a := .list (numbers (bucket processed (value / 100) ++ [value])))
        (List.getElem?_eq_some_iff.mp found).1
    · have separate : address ≠ other := by
        intro equal
        subst other
        exact same (store.exclusive key (value / 100) address otherLooked looked)
      have different : value / 100 ≠ key := Ne.symm same
      simpa [bucket, different, List.getElem?_set_ne separate] using otherFound
  · intro key absent
    have different : value / 100 ≠ key := by
      intro equal
      subst key
      rw [looked] at absent
      cases absent
    simpa [bucket, different] using store.missing key absent

/-- Search callers need the bucket contents after the construction frame has returned. -/
structure Buckets (heap : Mutable.Heap) (address : Nat) (values : List Int) where
  entries : List (Int × Nat)
  table : heap[address]? = some (.dict (encode entries))
  present : ∀ key target, lookup entries key = some target →
    address < target ∧ heap[target]? = some (.list (numbers (bucket values key)))
  missing : ∀ key, lookup entries key = none → bucket values key = []

def Store.view (store : Store base heap processed) : Buckets heap base.length processed where
  entries := store.entries
  table := store.table
  present := store.present
  missing := store.missing

variable {address : Nat} {values : List Int}

def Buckets.append (view : Buckets heap address values) (extra : Mutable.Heap) :
    Buckets (heap ++ extra) address values where
  entries := view.entries
  table := found_append heap extra address _ view.table
  present := by
    intro key target looked
    obtain ⟨later, found⟩ := view.present key target looked
    exact ⟨later, found_append heap extra target _ found⟩
  missing := view.missing

/-- Storing an outer dictionary entry cannot change a later nested prefix map. -/
def Buckets.set_before (view : Buckets heap address values) (location : Nat) (earlier : location < address)
    (object : Mutable.Object) : Buckets (heap.set location object) address values where
  entries := view.entries
  table := by simpa only [List.getElem?_set_ne (by omega : location ≠ address)] using view.table
  present := by
    intro key target looked
    obtain ⟨later, found⟩ := view.present key target looked
    exact ⟨later, by simpa only [List.getElem?_set_ne (by omega : location ≠ target)] using found⟩
  missing := view.missing

private def defaultExpression : Expr :=
  .call (.attribute (.name "mp") "setdefault") [.name "pref", .list []]

private theorem default_evaluates (store : Store base heap processed) (key : Int) (env : HeapCalls.Env)
    (getMap : env "mp" = some (.ref base.length)) (getKey : env "pref" = some (number key))
    (globals : Mutable.Globals) :
    ∃ (middle : Mutable.Heap) (address : Nat) (ready : Store base middle processed),
      lookup ready.entries key = some address ∧
      Evaluates program env defaultExpression heap (.ref address) middle globals := by
  have table := found_append heap [.list []] base.length _ store.table
  cases old : lookup store.entries key with
  | none =>
      let middle : Mutable.Heap := (heap ++ [Mutable.Object.list []]).set base.length
        (.dict (encode (store.entries ++ [(key, heap.length)])))
      let ready := store.add_empty key old
      refine ⟨middle, heap.length, ready, ?_, ?_⟩
      · change lookup (store.entries ++ [(key, heap.length)]) key = some heap.length
        simp [lookup_append, old, lookup]
      · apply Evaluates.dictionary_setdefault (address := base.length) (afterReceiver := heap)
          (afterArguments := heap ++ [.list []]) (values := [number key, .ref heap.length])
          (by intro name same; cases same; rw [getMap]; simp) (Evaluates.lookup getMap)
        · simp [bindMethod, store.table]
        · exact .cons (Evaluates.lookup getKey) (.cons (Evaluates.list .nil) .nil)
        · change Mutable.dictionarySetDefault base.length (number key) (.ref heap.length)
            (heap ++ [.list []]) = _
          rw [Mutable.dictionarySetDefault_scalar _ _ _ table (.int key) (.ref heap.length), lookup_encode, old]
          simp [middle, insert_encode _ _ _ old]
  | some address =>
      let ready := store.append [.list []]
      refine ⟨heap ++ [.list []], address, ready, old, ?_⟩
      apply Evaluates.dictionary_setdefault (address := base.length) (afterReceiver := heap)
        (afterArguments := heap ++ [.list []]) (values := [number key, .ref heap.length])
        (by intro name same; cases same; rw [getMap]; simp) (Evaluates.lookup getMap)
      · simp [bindMethod, store.table]
      · exact .cons (Evaluates.lookup getKey) (.cons (Evaluates.list .nil) .nil)
      · change Mutable.dictionarySetDefault base.length (number key) (.ref heap.length)
          (heap ++ [.list []]) = _
        rw [Mutable.dictionarySetDefault_scalar _ _ _ table (.int key) (.ref heap.length), lookup_encode, old]
        rfl

private def appendCode : List Stmt :=
  [.expr (.call (.attribute defaultExpression "append") [.name "x"])]

private theorem emit (store : Store base heap processed) (value : Int) (env : HeapCalls.Env)
    (getMap : env "mp" = some (.ref base.length)) (getKey : env "pref" = some (number (value / 100)))
    (getValue : env "x" = some (number value)) (globals : Mutable.Globals) :
    ∃ (finalHeap : Mutable.Heap) (_finalStore : Store base finalHeap (processed ++ [value])),
      BlockContinues program env appendCode heap env finalHeap globals := by
  obtain ⟨middle, address, ready, looked, prepared⟩ := default_evaluates store (value / 100) env getMap getKey globals
  have found := (ready.present (value / 100) address looked).2
  refine ⟨_, ready.bump value address looked, ?_⟩
  simpa only [numbers, List.map_append, List.map_cons, List.map_nil] using
    BlockContinues.discard (Evaluates.append prepared found (Evaluates.lookup getValue) found)

private def body : List Stmt :=
  [.assign [.name "pref"] (.binOp (.name "x") .floorDiv (.constant (.int 100)))] ++ appendCode
private def annotation : Expr :=
  .subscript (.name "dict") (.tuple [.name "int", .subscript (.name "list") (.name "int")])
private def code : List Stmt :=
  [.annAssign (.name "mp") annotation (some (.dict [])),
   .forIn (.name "x") (.name "nums") body [], .return (.name "mp")]

theorem code_source :
    executableBody ((findFunctionIn program.body "build_prefix_map").getD ([], [])).2 = code := rfl

private theorem source_found (store : Store base heap processed) (address : Nat) (values : List Int)
    (found : base[address]? = some (.list (numbers values))) :
    heap[address]? = some (.list (numbers values)) := by
  obtain ⟨extra, rfl⟩ := store.extension
  exact found_append base extra address _ found

/-- The actual source walks the original list and appends each number to its prefix bucket. -/
theorem loop (remaining values : List Int) (position : Nat) (suffix : values.drop position = remaining)
    (store : Store base heap processed) (input : Nat) (found : base[input]? = some (.list (numbers values)))
    (env : HeapCalls.Env) (getMap : env "mp" = some (.ref base.length)) (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_finalStore : Store base finalHeap (processed ++ remaining)),
      BlockReturns program env
        [.sequenceStepLoop "x" (.ref input) position body [], .return (.name "mp")]
        heap (.ref base.length) finalHeap globals := by
  induction remaining generalizing position heap processed env with
  | nil =>
      refine ⟨heap, ?_, ?_⟩
      · simpa only [List.append_nil] using store
      · apply BlockReturns.sequence_done (items := numbers values)
          (by simp [sequenceValue, Mutable.sequence, source_found store input values found])
        · have done : values[position]? = none := by rw [← List.head?_drop, suffix]; rfl
          simp [numbers, List.getElem?_map, done]
        · exact BlockReturns.return_value (Evaluates.lookup getMap)
  | cons value rest ih =>
      let bound := setLocal env "x" (number value)
      let prefixed := setLocal bound "pref" (number (value / 100))
      have boundMap : bound "mp" = some (.ref base.length) := by simpa [bound, setLocal] using getMap
      have prefixedMap : prefixed "mp" = some (.ref base.length) := by simpa [prefixed, setLocal] using boundMap
      have prefixRun : BlockContinues program bound
          [.assign [.name "pref"] (.binOp (.name "x") .floorDiv (.constant (.int 100)))] heap prefixed heap globals := by
        apply BlockContinues.assign
        apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
          (av := number value) (bv := number 100) (Evaluates.lookup rfl) Evaluates.constant
        simp [Mutable.binary, scalarBinary, number]
      obtain ⟨middle, ready, emitted⟩ := emit store value prefixed prefixedMap rfl
        (by simp [prefixed, bound, setLocal]) globals
      obtain ⟨finalHeap, finishedStore, executed⟩ := ih (position + 1)
        (by simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]) ready prefixed prefixedMap
      refine ⟨finalHeap, ?_, ?_⟩
      · simpa only [List.append_assoc, List.singleton_append] using finishedStore
      · apply BlockReturns.sequence_step (items := numbers values) (value := number value)
          (by simp [sequenceValue, Mutable.sequence, source_found store input values found])
        · have item : values[position]? = some value := by rw [← List.head?_drop, suffix]; rfl
          simp [numbers, List.getElem?_map, item]
        · simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, appendCode,
            List.append_assoc] using (prefixRun.append emitted).returns executed

/-- Arbitrary integer inputs produce exactly their prefix buckets in fresh heap objects. -/
theorem result (values : List Int) (heap : Mutable.Heap) (input : Nat)
    (found : heap[input]? = some (.list (numbers values))) (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_finalStore : Store heap finalHeap values),
      ReturnsValue program "build_prefix_map" [.ref input] heap (.ref heap.length) finalHeap globals := by
  let params := ((findFunctionIn program.body "build_prefix_map").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "build_prefix_map").getD ([], [])).2
  let env := functionEnv params sourceBody [.ref input] globals
  let initialized := setLocal env "mp" (.ref heap.length)
  let initial := Store.initial heap
  obtain ⟨finalHeap, finalStore, executed⟩ := loop values values 0 rfl initial input found initialized rfl globals
  refine ⟨finalHeap, by simpa using finalStore, ?_⟩
  apply BlockReturns.function rfl rfl
  change BlockReturns program env (executableBody sourceBody) heap (.ref heap.length) finalHeap globals
  rw [code_source]
  have allocated : BlockContinues program env [.annAssign (.name "mp") annotation (some (.dict []))]
      heap initialized (heap ++ [.dict []]) globals := BlockContinues.annAssign (Evaluates.dictionary .nil)
  apply allocated.returns
  apply BlockReturns.for_sequence (items := numbers values) (by intros; intro equal; cases equal)
    (Evaluates.lookup (by cases globals <;> rfl))
    (by simp [Mutable.sequence, source_found initial input values found])
  exact executed

end solvers.«61».build_prefix_map

namespace solvers.«61».solve_setup
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

private abbrev encode := build_prefix_map.encode
private abbrev lookup := build_prefix_map.lookup
private abbrev Buckets := build_prefix_map.Buckets

/-- The outer prefix dictionary owns one correct nested map for every completed family. -/
structure PrefixState (base heap : Mutable.Heap) (processed : List Int) where
  entries : List (Int × Nat)
  keys : entries.map Prod.fst = processed
  extension : ∃ extra, heap = base ++ extra
  table : heap[base.length]? = some (.dict (encode entries))
  children : ∀ key address, (key, address) ∈ entries →
    base.length < address ∧ Nonempty (Buckets heap address (candidates key))

def PrefixState.initial (base : Mutable.Heap) : PrefixState base (base ++ [.dict []]) [] where
  entries := []
  keys := rfl
  extension := ⟨[.dict []], rfl⟩
  table := by simp [encode, build_prefix_map.encode]
  children := by intro key address member; simp at member

variable {base heap : Mutable.Heap} {processed : List Int}

def PrefixState.append (world : PrefixState base heap processed) (extra : Mutable.Heap) :
    PrefixState base (heap ++ extra) processed where
  entries := world.entries
  keys := world.keys
  extension := by
    obtain ⟨old, rfl⟩ := world.extension
    exact ⟨old ++ extra, List.append_assoc _ _ _⟩
  table := build_prefix_map.found_append heap extra _ _ world.table
  children := by
    intro key address member
    obtain ⟨later, ⟨view⟩⟩ := world.children key address member
    exact ⟨later, ⟨view.append extra⟩⟩

variable {childHeap : Mutable.Heap}

def PrefixState.insert_child (world : PrefixState base heap processed) (sides : Int)
    (child : build_prefix_map.Store heap childHeap (candidates sides)) :
    PrefixState base (childHeap.set base.length (.dict (encode (world.entries ++ [(sides, heap.length)]))))
      (processed ++ [sides]) := by
  have inside : base.length < heap.length := (List.getElem?_eq_some_iff.mp world.table).1
  have larger : heap.length ≤ childHeap.length := by
    obtain ⟨extra, extension⟩ := child.extension
    simp [extension]
  refine { entries := world.entries ++ [(sides, heap.length)]
           keys := by simp [world.keys]
           extension := ?_
           table := ?_
           children := ?_ }
  · have combined : ∃ extra, childHeap = base ++ extra := by
      obtain ⟨extra, extension⟩ := child.extension
      obtain ⟨old, before⟩ := world.extension
      exact ⟨old ++ extra, by rw [extension, before, List.append_assoc]⟩
    exact build_prefix_map.extension_set base childHeap combined _ (Nat.le_refl _) _
  · exact List.getElem?_set_self (by omega)
  · intro key address member
    obtain ⟨extra, extension⟩ := child.extension
    rcases List.mem_append.mp member with old | fresh
    · obtain ⟨later, ⟨view⟩⟩ := world.children key address old
      have extended : Buckets childHeap address (candidates key) := by
        rw [extension]
        exact view.append extra
      exact ⟨later, ⟨extended.set_before base.length later _⟩⟩
    · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.mp fresh)
      exact ⟨inside, ⟨child.view.set_before base.length inside _⟩⟩

private def prefixValue : Expr :=
  .call (.name "build_prefix_map") [.subscript (.name "nums_by_type") (.name "t")]

private def prefixBody (collector : Nat) : List Stmt :=
  [.comprehensionStore collector (.name "t") prefixValue]

/-- One real callback builds its nested map, then stores that reference in the outer collector. -/
theorem prefix_store (caller : Mutable.Heap) (world : PrefixState (generatedHeap caller) heap processed)
    (sides : Int) (lower : 3 ≤ sides) (upper : sides ≤ 8) (absent : lookup world.entries sides = none)
    (env : HeapCalls.Env) (getNumbers : env "nums_by_type" = some (.ref (initialHeap caller).length))
    (getType : env "t" = some (number sides)) (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_next : PrefixState (generatedHeap caller) finalHeap (processed ++ [sides])),
      BlockContinues program env (prefixBody (generatedHeap caller).length) heap env finalHeap globals := by
  obtain ⟨before, extension⟩ := world.extension
  have table : heap[(initialHeap caller).length]? = some (.dict (numberEntries caller)) := by
    rw [extension]
    exact build_prefix_map.found_append _ _ _ _ (generated_table caller)
  have found : heap[caller.length + sides.toNat]? = some (.list (numbers (candidates sides))) := by
    rw [extension]
    exact build_prefix_map.found_append _ _ _ _ (generated_family caller sides lower upper)
  have argument : Evaluates program env (.subscript (.name "nums_by_type") (.name "t")) heap
      (.ref (caller.length + sides.toNat)) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getNumbers) (Evaluates.lookup getType)
    exact Mutable.getItem_dictionary _ _ _ table (.int sides) (numberEntries_lookup caller sides lower upper)
  obtain ⟨childHeap, child, returned⟩ := build_prefix_map.result (candidates sides) heap
    (caller.length + sides.toNat) found globals
  have evaluated : Evaluates program env prefixValue heap (.ref heap.length) childHeap globals :=
    Evaluates.call (.cons argument .nil) returned
  have collector : childHeap[(generatedHeap caller).length]? = some (.dict (encode world.entries)) := by
    obtain ⟨extra, extension⟩ := child.extension
    rw [extension]
    exact build_prefix_map.found_append _ _ _ _ world.table
  refine ⟨_, world.insert_child sides child, ?_⟩
  apply BlockContinues.comprehension_store (.cons (Evaluates.lookup getType) (.cons evaluated .nil))
  have stored := Mutable.setItem_dictionary _ _ _ collector (.int sides) (.ref heap.length)
  simpa only [number, build_prefix_map.insert_encode world.entries sides heap.length absent] using stored

private theorem lookup_missing (entries : List (Int × Nat)) (key : Int)
    (absent : key ∉ entries.map Prod.fst) : lookup entries key = none := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
      obtain ⟨stored, address⟩ := entry
      have different : stored ≠ key := by intro equal; simp [equal] at absent
      have tail : key ∉ rest.map Prod.fst := by simpa [different, Ne.symm different] using absent
      simpa [lookup, build_prefix_map.lookup, different] using ih tail

/-- The source comprehension builds all remaining maps without changing earlier families. -/
theorem prefix_loop (caller : Mutable.Heap) (remaining : List Int) (position : Nat)
    (suffix : types.drop position = remaining) (distinct : remaining.Nodup)
    (fresh : ∀ key ∈ remaining, key ∉ processed)
    (valid : ∀ key ∈ remaining, (3 : Int) ≤ key ∧ key ≤ 8)
    (world : PrefixState (generatedHeap caller) heap processed)
    (env : HeapCalls.Env) (getNumbers : env "nums_by_type" = some (.ref (initialHeap caller).length))
    (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (finalEnv : HeapCalls.Env)
      (_next : PrefixState (generatedHeap caller) finalHeap (processed ++ remaining)),
      BlockContinues program env
        [.sequenceStepLoop "t" (.ref caller.length) position (prefixBody (generatedHeap caller).length) []]
        heap finalEnv finalHeap globals := by
  have typesFound : (generatedHeap caller)[caller.length]? = some (.list (numbers types)) := by
    simp [generated_layout, initialHeap, List.append_assoc]
  induction remaining generalizing position heap processed env with
  | nil =>
      refine ⟨heap, env, by simpa using world, ?_⟩
      apply BlockContinues.sequence_done (items := numbers types)
      · obtain ⟨extra, extension⟩ := world.extension
        have found := build_prefix_map.found_append _ extra _ _ typesFound
        rw [← extension] at found
        simp [sequenceValue, Mutable.sequence, found]
      · have done : types[position]? = none := by rw [← List.head?_drop, suffix]; rfl
        simp [numbers, List.getElem?_map, done]
      · exact .nil
  | cons sides rest ih =>
      let bound := setLocal env "t" (number sides)
      have boundNumbers : bound "nums_by_type" = some (.ref (initialHeap caller).length) := by
        simpa [bound, setLocal] using getNumbers
      have absent : lookup world.entries sides = none := lookup_missing world.entries sides
        (by rw [world.keys]; exact fresh sides (by simp))
      have bounds := valid sides (by simp)
      obtain ⟨middle, ready, emitted⟩ := prefix_store caller world sides bounds.1 bounds.2 absent bound boundNumbers rfl globals
      obtain ⟨finalHeap, finalEnv, finished, executed⟩ := ih (position + 1)
        (by simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons])
        (List.nodup_cons.mp distinct).2
        (by
          intro key member
          have old := fresh key (by simp [member])
          have different : key ≠ sides := by
            intro equal
            subst key
            exact (List.nodup_cons.mp distinct).1 member
          simp [old, different])
        (by intro key member; exact valid key (by simp [member])) ready bound boundNumbers
      refine ⟨finalHeap, finalEnv, ?_, ?_⟩
      · simpa only [List.append_assoc, List.singleton_append] using finished
      · apply BlockContinues.sequence_step (items := numbers types) (value := number sides)
        · obtain ⟨extra, extension⟩ := world.extension
          have found := build_prefix_map.found_append _ extra _ _ typesFound
          rw [← extension] at found
          simp [sequenceValue, Mutable.sequence, found]
        · have item : types[position]? = some sides := by rw [← List.head?_drop, suffix]; rfl
          simp [numbers, List.getElem?_map, item]
        · simpa [prefixBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using emitted.append executed

private def prefixGenerators : List (Expr × Expr × List Expr) := [(.name "t", typeCell, [])]
private def prefixComprehension : Expr := .dictComp (.name "t") prefixValue prefixGenerators

theorem prefixes_evaluate (caller : Mutable.Heap) (env : HeapCalls.Env)
    (getCell : env "__formalsw_cell_solve_types" = some (.ref (caller.length + 1)))
    (getNumbers : env "nums_by_type" = some (.ref (initialHeap caller).length))
    (globals : Mutable.Globals := []) :
    ∃ (finalHeap : Mutable.Heap) (_world : PrefixState (generatedHeap caller) finalHeap types),
      Evaluates program env prefixComprehension (generatedHeap caller)
        (.ref (generatedHeap caller).length) finalHeap globals := by
  let masked := comprehensionTargetsEnv env prefixGenerators
  have maskedNumbers : masked "nums_by_type" = some (.ref (initialHeap caller).length) := by
    simpa [masked, prefixGenerators, comprehensionTargetsEnv, targetLocals] using getNumbers
  obtain ⟨finalHeap, finalEnv, world, executed⟩ := prefix_loop caller types 0 rfl (by decide)
    (by intro key _; simp) (by intro key member; simp [types] at member; omega)
    (PrefixState.initial (generatedHeap caller)) masked maskedNumbers globals
  refine ⟨finalHeap, by simpa using world, ?_⟩
  have cellFound : (generatedHeap caller)[caller.length + 1]? = some (.list [.ref caller.length]) := by
    simp [generated_layout, initialHeap, List.append_assoc]
  have typesFound : (generatedHeap caller)[caller.length]? = some (.list (numbers types)) := by
    simp [generated_layout, initialHeap, List.append_assoc]
  have iterable : Evaluates program env typeCell (generatedHeap caller) (.ref caller.length)
      (generatedHeap caller) globals := by
    apply Evaluates.subscript (middle := generatedHeap caller) (afterIndex := generatedHeap caller)
      (finalHeap := generatedHeap caller) (by intros; intro impossible; cases impossible)
      (Evaluates.lookup getCell) Evaluates.constant
    exact Mutable.getItem_list _ _ 0 [.ref caller.length] cellFound (by simp)
  apply Evaluates.dict_comp_sequence (receiver := .ref caller.length) (items := numbers types)
    (middle := generatedHeap caller) (finalEnv := finalEnv)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    iterable (by simp [Mutable.sequence, typesFound])
  simpa only [prefixGenerators, comprehensionTargetName, comprehensionTargetBody, comprehensionFilters,
    List.foldr_nil, comprehensionTargetsBody, sequenceSource, masked, prefixBody] using executed

def prefixCode : List Stmt :=
  [.assign [.name "__formalsw_cell_solve_pref_map"] (.list [prefixComprehension])]

theorem prefix_source :
    (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).take 3 =
      initialCode ++ generateCode ++ prefixCode := rfl

def Prepared (caller : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals) : Prop :=
    ∃ (finalHeap : Mutable.Heap) (cell : Nat),
      Nonempty (PrefixState (generatedHeap caller) finalHeap types) ∧
      BlockContinues program env
        ((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).take 3)
        caller (setLocal (generatedEnv env caller) "__formalsw_cell_solve_pref_map" (.ref cell)) finalHeap globals ∧
      finalHeap[cell]? = some (.list [.ref (generatedHeap caller).length])

/-- The complete source setup creates the candidate tables and both closure cells. -/
theorem prepared_result (caller : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals := []) :
    Prepared caller env globals := by
  obtain ⟨prefixHeap, world, evaluated⟩ := prefixes_evaluate caller (generatedEnv env caller)
    (by simp [generatedEnv, initialEnv, setLocal]) rfl globals
  refine ⟨prefixHeap ++ [.list [.ref (generatedHeap caller).length]], prefixHeap.length,
    ⟨world.append [.list [.ref (generatedHeap caller).length]]⟩, ?_, by simp⟩
  rw [prefix_source]
  apply (generate_result caller env globals).append
  exact BlockContinues.assign (Evaluates.list (.cons evaluated .nil))

end solvers.«61».solve_setup

namespace solvers.«61»

theorem prefix_setup_source_result (caller : FormalSW.PyAst.Mutable.Heap)
    (env : FormalSW.PyAst.HeapCalls.Env) (globals : FormalSW.PyAst.Mutable.Globals) :
    solve_setup.Prepared caller env globals := solve_setup.prepared_result caller env globals

theorem build_prefix_map_source_result (values : List Int) (heap : FormalSW.PyAst.Mutable.Heap) (input : Nat)
    (globals : FormalSW.PyAst.Mutable.Globals) :
    heap[input]? = some (.list (values.map (fun value => .scalar (.int value)))) →
    ∃ finalHeap, Nonempty (build_prefix_map.Store heap finalHeap values) ∧
      FormalSW.PyAst.HeapCalls.ReturnsValue Solvers.P61Ast.module "build_prefix_map"
        [.ref input] heap (.ref heap.length) finalHeap globals := by
  intro found
  obtain ⟨finalHeap, store, returned⟩ := build_prefix_map.result values heap input found globals
  exact ⟨finalHeap, ⟨store⟩, returned⟩

end solvers.«61»

namespace solvers.«61».dfs
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

private def integerSet (items : List Int) : Mutable.Object := .set (items.map Constant.int)

def extendedSet (setName itemName : String) : Expr :=
  .binOp (.name setName) .bitOr (.builtinCall .setDisplay [.name itemName])

/-- The recursive call's set-union argument preserves its parent set and allocates two objects. -/
theorem grow_set (heap : Mutable.Heap) (address : Nat) (items : List Int) (value : Int)
    (fresh : value ∉ items) (found : heap[address]? = some (integerSet items))
    (setName itemName : String) (env : HeapCalls.Env)
    (getSet : env setName = some (.ref address)) (getItem : env itemName = some (number value))
    (globals : Mutable.Globals := []) :
    Evaluates program env (extendedSet setName itemName) heap (.ref (heap.length + 1))
      (heap ++ [.set [.int value], integerSet (items ++ [value])]) globals := by
  have literal : Evaluates program env (.builtinCall .setDisplay [.name itemName]) heap (.ref heap.length)
      (heap ++ [.set [.int value]]) globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getItem) .nil)
    exact Mutable.allocateSetValues_singleton heap (.int value)
  have left : (heap ++ [.set [.int value]])[address]? = some (integerSet items) :=
    build_prefix_map.found_append _ _ _ _ found
  apply Evaluates.binary (middle := heap) (afterRight := heap ++ [.set [.int value]])
    (Evaluates.lookup getSet) literal
  have operation := Mutable.binary_union_sets (heap ++ [.set [.int value]]) address heap.length
    (items.map Constant.int) [.int value] left (by simp)
  simpa only [ScalarSet.union_singleton_ints items value fresh, integerSet, List.length_append,
    List.length_cons, List.length_nil, Nat.add_zero, List.append_assoc, List.singleton_append] using operation

private abbrev Node := ProjectEulerSolutions.PolygonalCycle.Node

def references (nodes : List (Nat × Node)) : List Mutable.Value := nodes.map (fun node => .ref node.1)

/-- A path retains the actual tuple objects together with their numerical model. -/
structure PathView (heap : Mutable.Heap) (address : Nat) (nodes : List (Nat × Node)) where
  found : heap[address]? = some (.list (references nodes))
  contents : ∀ target item, (target, item) ∈ nodes →
    heap[target]? = some (.tuple [number item.1, number item.2])

variable {heap : Mutable.Heap} {address : Nat} {nodes : List (Nat × Node)}

def PathView.append (view : PathView heap address nodes) (extra : Mutable.Heap) :
    PathView (heap ++ extra) address nodes where
  found := build_prefix_map.found_append _ _ _ _ view.found
  contents := by
    intro target item member
    exact build_prefix_map.found_append _ _ _ _ (view.contents target item member)

def pathHeap (heap : Mutable.Heap) (nodes : List (Nat × Node)) (value kind : Int) : Mutable.Heap :=
  heap ++ [.tuple [number value, number kind], .list [.ref heap.length],
    .list (references nodes ++ [.ref heap.length])]

def extendedPath : Expr :=
  .binOp (.name "path") .add (.list [.tuple [.name "nxt", .name "t"]])

/-- Python's concatenation creates a tuple, a temporary singleton list, and a new path list. -/
theorem grow_path (view : PathView heap address nodes) (value kind : Int) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref address)) (getValue : env "nxt" = some (number value))
    (getKind : env "t" = some (number kind)) (globals : Mutable.Globals := []) :
    Evaluates program env extendedPath heap (.ref (heap.length + 2)) (pathHeap heap nodes value kind) globals := by
  have tuple : Evaluates program env (.tuple [.name "nxt", .name "t"]) heap (.ref heap.length)
      (heap ++ [.tuple [number value, number kind]]) globals :=
    Evaluates.tuple (.cons (Evaluates.lookup getValue) (.cons (Evaluates.lookup getKind) .nil)) (by simp)
  have singleton : Evaluates program env (.list [.tuple [.name "nxt", .name "t"]]) heap (.ref (heap.length + 1))
      (heap ++ [.tuple [number value, number kind], .list [.ref heap.length]]) globals := by
    convert Evaluates.list (.cons tuple .nil) using 1 <;> simp [List.length_append, List.append_assoc]
  have left : (heap ++ [.tuple [number value, number kind], .list [.ref heap.length]])[address]? =
      some (.list (references nodes)) := by
    exact build_prefix_map.found_append heap
      [.tuple [number value, number kind], .list [.ref heap.length]] address _ view.found
  have operation : Mutable.binary .add (.ref address) (.ref (heap.length + 1))
      (heap ++ [.tuple [number value, number kind], .list [.ref heap.length]]) =
      some (.ok (.ref (heap.length + 2)), pathHeap heap nodes value kind) := by
    simp [Mutable.binary, show (BinOp.add == BinOp.add) = true from rfl, Mutable.array, left,
      Mutable.allocate, Except.map, pathHeap, List.append_assoc]
  apply Evaluates.binary (middle := heap)
    (afterRight := heap ++ [.tuple [number value, number kind], .list [.ref heap.length]])
    (Evaluates.lookup getPath) singleton
  simpa only [pathHeap, List.length_append, List.length_cons, List.length_nil, Nat.add_zero,
    List.append_assoc, List.singleton_append] using operation

def PathView.grown (view : PathView heap address nodes) (value kind : Int) :
    PathView (pathHeap heap nodes value kind) (heap.length + 2) (nodes ++ [(heap.length, (value, kind))]) where
  found := by simp [pathHeap, references]
  contents := by
    intro target item member
    rcases List.mem_append.mp member with old | fresh
    · exact build_prefix_map.found_append _ _ _ _ (view.contents target item old)
    · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.mp fresh)
      simp [pathHeap]

def usedKinds (nodes : List (Nat × Node)) : List Int := nodes.map (fun node => node.2.2)
def usedNumbers (nodes : List (Nat × Node)) : List Int := nodes.map (fun node => node.2.1)

def argumentHeap (heap : Mutable.Heap) (nodes : List (Nat × Node)) (value kind : Int) : Mutable.Heap :=
  pathHeap heap nodes value kind ++
    [.set [.int kind], integerSet (usedKinds nodes ++ [kind]),
     .set [.int value], integerSet (usedNumbers nodes ++ [value])]

def recursiveArguments : List Expr :=
  [extendedPath, extendedSet "used_types" "t", extendedSet "used_nums" "nxt",
   .name "__formalsw_cell_solve_pref_map", .name "__formalsw_cell_solve_types"]
def recursiveCall : Expr := .call (.name "__formalsw_local_solve_dfs") recursiveArguments
def resultCondition : Expr := .unaryOp .not (.builtinCall .identity [.name "res", .constant .none])
def nextValues : Expr :=
  .call (.attribute (.subscript
    (.subscript (.name "__formalsw_cell_solve_pref_map") (.constant (.int 0))) (.name "t")) "get")
    [.name "suffix", .list []]
def innerBody : List Stmt :=
  [.ifThen (.membership (.name "nxt") (.name "used_nums") false) [.continueLoop] [],
   .assign [.name "res"] recursiveCall,
   .ifThen resultCondition [.return (.name "res")] []]
def outerBody : List Stmt :=
  [.ifThen (.membership (.name "t") (.name "used_types") false) [.continueLoop] [],
   .forIn (.name "nxt") nextValues innerBody []]
def code : List Stmt :=
  [.ifThen condition closedBody [],
   .assign [.name "suffix"] (.binOp lastExpression .mod (.constant (.int 100))),
   .forIn (.name "t") solve_setup.typeCell outerBody [], .return (.constant .none)]

theorem code_source :
    executableBody ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2 = code := rfl

/-- The source prepares all five recursive arguments in order, preserving every parent object. -/
theorem arguments_evaluate (view : PathView heap address nodes) (value kind : Int)
    (typesAddress numbersAddress : Nat)
    (typesFound : heap[typesAddress]? = some (integerSet (usedKinds nodes)))
    (numbersFound : heap[numbersAddress]? = some (integerSet (usedNumbers nodes)))
    (freshKind : kind ∉ usedKinds nodes) (freshValue : value ∉ usedNumbers nodes)
    (env : HeapCalls.Env) (prefixCell typesCell : Mutable.Value)
    (getPath : env "path" = some (.ref address))
    (getTypes : env "used_types" = some (.ref typesAddress))
    (getNumbers : env "used_nums" = some (.ref numbersAddress))
    (getValue : env "nxt" = some (number value)) (getKind : env "t" = some (number kind))
    (getPrefixCell : env "__formalsw_cell_solve_pref_map" = some prefixCell)
    (getTypesCell : env "__formalsw_cell_solve_types" = some typesCell)
    (globals : Mutable.Globals := []) :
    Evaluates.Arguments program env recursiveArguments heap
      [.ref (heap.length + 2), .ref (heap.length + 4), .ref (heap.length + 6), prefixCell, typesCell]
      (argumentHeap heap nodes value kind) globals := by
  let pathReady := pathHeap heap nodes value kind
  let typesReady := pathReady ++ [.set [.int kind], integerSet (usedKinds nodes ++ [kind])]
  have typesPreserved : pathReady[typesAddress]? = some (integerSet (usedKinds nodes)) :=
    build_prefix_map.found_append _ _ _ _ typesFound
  have numbersPreserved : typesReady[numbersAddress]? = some (integerSet (usedNumbers nodes)) := by
    have first : pathReady[numbersAddress]? = some (integerSet (usedNumbers nodes)) :=
      build_prefix_map.found_append _ _ _ _ numbersFound
    exact build_prefix_map.found_append _ _ _ _ first
  have typeRun : Evaluates program env (extendedSet "used_types" "t") pathReady (.ref (heap.length + 4))
      typesReady globals := by
    simpa [typesReady, pathReady, pathHeap, List.length_append, List.append_assoc, Nat.add_assoc] using
      grow_set pathReady typesAddress (usedKinds nodes) kind freshKind typesPreserved "used_types" "t"
        env getTypes getKind globals
  have numberRun : Evaluates program env (extendedSet "used_nums" "nxt") typesReady (.ref (heap.length + 6))
      (argumentHeap heap nodes value kind) globals := by
    simpa [typesReady, pathReady, pathHeap, argumentHeap, List.length_append, List.append_assoc, Nat.add_assoc] using
      grow_set typesReady numbersAddress (usedNumbers nodes) value freshValue numbersPreserved "used_nums" "nxt"
        env getNumbers getValue globals
  exact .cons (grow_path view value kind env getPath getValue getKind globals)
    (.cons typeRun (.cons numberRun (.cons (Evaluates.lookup getPrefixCell) (.cons (Evaluates.lookup getTypesCell) .nil))))

def PathView.arguments (view : PathView heap address nodes) (value kind : Int) :
    PathView (argumentHeap heap nodes value kind) (heap.length + 2) (nodes ++ [(heap.length, (value, kind))]) :=
  (view.grown value kind).append
    [.set [.int kind], integerSet (usedKinds nodes ++ [kind]),
     .set [.int value], integerSet (usedNumbers nodes ++ [value])]

theorem argument_types (heap : Mutable.Heap) (nodes : List (Nat × Node)) (value kind : Int) :
    (argumentHeap heap nodes value kind)[heap.length + 4]? =
      some (integerSet (usedKinds (nodes ++ [(heap.length, (value, kind))]))) := by
  simp [argumentHeap, pathHeap, usedKinds, List.append_assoc]

theorem argument_numbers (heap : Mutable.Heap) (nodes : List (Nat × Node)) (value kind : Int) :
    (argumentHeap heap nodes value kind)[heap.length + 6]? =
      some (integerSet (usedNumbers (nodes ++ [(heap.length, (value, kind))]))) := by
  simp [argumentHeap, pathHeap, usedNumbers, List.append_assoc]

private def numbers (items : List Int) : List Mutable.Value := items.map number
private abbrev candidates := ProjectEulerSolutions.PolygonalCycle.candidates

private theorem lookup_of_mem (entries : List (Int × Nat)) (key : Int) (address : Nat)
    (distinct : (entries.map Prod.fst).Nodup) (member : (key, address) ∈ entries) :
    build_prefix_map.lookup entries key = some address := by
  induction entries with
  | nil => simp at member
  | cons entry rest ih =>
      obtain ⟨stored, target⟩ := entry
      rcases List.mem_cons.mp member with same | later
      · obtain ⟨rfl, rfl⟩ := Prod.mk.inj same
        simp [build_prefix_map.lookup]
      · have different : stored ≠ key := by
          intro equal
          have present : key ∈ rest.map Prod.fst := List.mem_map.mpr ⟨(key, address), later, rfl⟩
          exact (List.nodup_cons.mp distinct).1 (by simpa [equal] using present)
        simpa [build_prefix_map.lookup, different] using ih (List.nodup_cons.mp distinct).2 later

/-- Captured read-only tables needed by every recursive source call. -/
structure World (heap : Mutable.Heap) (prefixCell typesCell : Nat) where
  typesAddress : Nat
  prefixAddress : Nat
  typesCellFound : heap[typesCell]? = some (.list [.ref typesAddress])
  typesFound : heap[typesAddress]? = some (.list (numbers solve_setup.types))
  prefixCellFound : heap[prefixCell]? = some (.list [.ref prefixAddress])
  family : ∀ kind ∈ solve_setup.types, ∃ target entries,
    heap[prefixAddress]? = some (.dict entries) ∧
    ScalarDict.lookup entries (.int kind) = some (.ref target) ∧
    Nonempty (build_prefix_map.Buckets heap target (candidates kind))

variable {prefixCell typesCell : Nat}

def World.append (world : World heap prefixCell typesCell) (extra : Mutable.Heap) :
    World (heap ++ extra) prefixCell typesCell where
  typesAddress := world.typesAddress
  prefixAddress := world.prefixAddress
  typesCellFound := build_prefix_map.found_append _ _ _ _ world.typesCellFound
  typesFound := build_prefix_map.found_append _ _ _ _ world.typesFound
  prefixCellFound := build_prefix_map.found_append _ _ _ _ world.prefixCellFound
  family := by
    intro kind member
    obtain ⟨target, entries, found, looked, ⟨buckets⟩⟩ := world.family kind member
    exact ⟨target, entries, build_prefix_map.found_append _ _ _ _ found, looked,
      ⟨buckets.append extra⟩⟩

def World.of_setup (caller : Mutable.Heap) (cell : Nat)
    (state : solve_setup.PrefixState (solve_setup.generatedHeap caller) heap solve_setup.types)
    (cellFound : heap[cell]? = some (.list [.ref (solve_setup.generatedHeap caller).length])) :
    World heap cell (caller.length + 1) where
  typesAddress := caller.length
  prefixAddress := (solve_setup.generatedHeap caller).length
  typesCellFound := by
    obtain ⟨extra, extension⟩ := state.extension
    rw [extension]
    apply build_prefix_map.found_append
    simp [solve_setup.generated_layout, solve_setup.initialHeap, List.append_assoc]
  typesFound := by
    obtain ⟨extra, extension⟩ := state.extension
    rw [extension]
    apply build_prefix_map.found_append
    simp [solve_setup.generated_layout, solve_setup.initialHeap, List.append_assoc, solve_setup.numbers, numbers]
  prefixCellFound := cellFound
  family := by
    intro kind member
    have inKeys : kind ∈ state.entries.map Prod.fst := by rw [state.keys]; exact member
    obtain ⟨⟨key, target⟩, present, equal⟩ := List.mem_map.mp inKeys
    change key = kind at equal
    subst key
    have distinct : (state.entries.map Prod.fst).Nodup := by rw [state.keys]; decide
    have looked := lookup_of_mem state.entries kind target distinct present
    obtain ⟨_, ⟨buckets⟩⟩ := state.children kind target present
    change build_prefix_map.Buckets heap target (generate_4digit_polygonals.answer kind) at buckets
    rw [generate_4digit_polygonals.candidates_eq_model kind (by simp [solve_setup.types] at member; omega)] at buckets
    refine ⟨target, build_prefix_map.encode state.entries, state.table, ?_, ⟨buckets⟩⟩
    simp only [build_prefix_map.lookup_encode, looked, Option.map_some]

theorem types_evaluate (world : World heap prefixCell typesCell) (env : HeapCalls.Env)
    (getCell : env "__formalsw_cell_solve_types" = some (.ref typesCell)) (globals : Mutable.Globals := []) :
    Evaluates program env solve_setup.typeCell heap (.ref world.typesAddress) heap globals := by
  apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
    (by intros; intro impossible; cases impossible) (Evaluates.lookup getCell) Evaluates.constant
  exact Mutable.getItem_list _ _ 0 [.ref world.typesAddress] world.typesCellFound (by simp)

/-- The real `.get(suffix, [])` allocates its default on hits as well as misses. -/
theorem next_values_evaluate (world : World heap prefixCell typesCell) (kind suffix : Int)
    (member : kind ∈ solve_setup.types) (env : HeapCalls.Env)
    (getCell : env "__formalsw_cell_solve_pref_map" = some (.ref prefixCell))
    (getKind : env "t" = some (number kind)) (getSuffix : env "suffix" = some (number suffix))
    (globals : Mutable.Globals := []) :
    ∃ result, Evaluates program env nextValues heap (.ref result) (heap ++ [.list []]) globals ∧
      (heap ++ [.list []])[result]? =
        some (.list (numbers (build_prefix_map.bucket (candidates kind) suffix))) := by
  obtain ⟨target, entries, table, looked, ⟨buckets⟩⟩ := world.family kind member
  have cell : Evaluates program env
      (.subscript (.name "__formalsw_cell_solve_pref_map") (.constant (.int 0))) heap
      (.ref world.prefixAddress) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getCell) Evaluates.constant
    exact Mutable.getItem_list _ _ 0 [.ref world.prefixAddress] world.prefixCellFound (by simp)
  have receiver : Evaluates program env
      (.subscript (.subscript (.name "__formalsw_cell_solve_pref_map") (.constant (.int 0))) (.name "t"))
      heap (.ref target) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) cell (Evaluates.lookup getKind)
    exact Mutable.getItem_dictionary _ _ _ table (.int kind) looked
  let extended := buckets.append [.list []]
  have sameEntries : extended.entries = buckets.entries := rfl
  cases selected : build_prefix_map.lookup buckets.entries suffix with
  | none =>
      refine ⟨heap.length, ?_, ?_⟩
      · apply Evaluates.dictionary_get (afterReceiver := heap) (afterArguments := heap ++ [.list []])
          (values := [number suffix, .ref heap.length])
          (by intro name impossible; cases impossible) receiver buckets.table
          (.cons (Evaluates.lookup getSuffix) (.cons (Evaluates.list .nil) .nil))
        change Mutable.dictionaryGet target (number suffix) (.ref heap.length) (heap ++ [.list []]) = _
        rw [Mutable.dictionaryGet_scalar _ _ _ extended.table (.int suffix) (.ref heap.length),
          build_prefix_map.lookup_encode, sameEntries, selected]
        rfl
      · rw [buckets.missing suffix selected]
        simp [numbers]
  | some result =>
      refine ⟨result, ?_, ?_⟩
      · apply Evaluates.dictionary_get (afterReceiver := heap) (afterArguments := heap ++ [.list []])
          (values := [number suffix, .ref heap.length])
          (by intro name impossible; cases impossible) receiver buckets.table
          (.cons (Evaluates.lookup getSuffix) (.cons (Evaluates.list .nil) .nil))
        change Mutable.dictionaryGet target (number suffix) (.ref heap.length) (heap ++ [.list []]) = _
        rw [Mutable.dictionaryGet_scalar _ _ _ extended.table (.int suffix) (.ref heap.length),
          build_prefix_map.lookup_encode, sameEntries, selected]
        rfl
      · exact build_prefix_map.found_append _ _ _ _ (buckets.present suffix result selected).2

/-- All five source parameters describe one logical search state. -/
structure SearchState (heap : Mutable.Heap) (path typesAddress numbersAddress prefixCell typesCell : Nat)
    (nodes : List (Nat × Node)) where
  world : World heap prefixCell typesCell
  pathView : PathView heap path nodes
  typesFound : heap[typesAddress]? = some (integerSet (usedKinds nodes))
  numbersFound : heap[numbersAddress]? = some (integerSet (usedNumbers nodes))

def SearchState.append (state : SearchState heap path typesAddress numbersAddress prefixCell typesCell nodes)
    (extra : Mutable.Heap) :
    SearchState (heap ++ extra) path typesAddress numbersAddress prefixCell typesCell nodes where
  world := state.world.append extra
  pathView := state.pathView.append extra
  typesFound := build_prefix_map.found_append _ _ _ _ state.typesFound
  numbersFound := build_prefix_map.found_append _ _ _ _ state.numbersFound

/-- The child uses fresh path and set objects while retaining its parent's captured tables. -/
def SearchState.arguments (state : SearchState heap path typesAddress numbersAddress prefixCell typesCell nodes)
    (value kind : Int) :
    SearchState (argumentHeap heap nodes value kind) (heap.length + 2) (heap.length + 4)
      (heap.length + 6) prefixCell typesCell (nodes ++ [(heap.length, (value, kind))]) where
  world := by
    simpa only [argumentHeap, pathHeap, List.append_assoc, List.singleton_append] using state.world.append
      [.tuple [number value, number kind], .list [.ref heap.length],
       .list (references nodes ++ [Mutable.Value.ref heap.length]),
       .set [.int kind], integerSet (usedKinds nodes ++ [kind]),
       .set [.int value], integerSet (usedNumbers nodes ++ [value])]
  pathView := state.pathView.arguments value kind
  typesFound := argument_types heap nodes value kind
  numbersFound := argument_numbers heap nodes value kind

/-- Local loop variables do not replace the recursive call's five parameters. -/
structure Bindings (env : HeapCalls.Env) (path typesAddress numbersAddress prefixCell typesCell : Nat) : Prop where
  getPath : env "path" = some (.ref path)
  getTypes : env "used_types" = some (.ref typesAddress)
  getNumbers : env "used_nums" = some (.ref numbersAddress)
  getPrefixCell : env "__formalsw_cell_solve_pref_map" = some (.ref prefixCell)
  getTypesCell : env "__formalsw_cell_solve_types" = some (.ref typesCell)

theorem Bindings.local (bound : Bindings env path typesAddress numbersAddress prefixCell typesCell)
    (name : String) (value : Mutable.Value)
    (fresh : name ∉ ["path", "used_types", "used_nums",
      "__formalsw_cell_solve_pref_map", "__formalsw_cell_solve_types"]) :
    Bindings (setLocal env name value) path typesAddress numbersAddress prefixCell typesCell := by
  simp only [List.mem_cons, not_or] at fresh
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
    simp_all [setLocal, eq_comm, bound.getPath, bound.getTypes, bound.getNumbers,
      bound.getPrefixCell, bound.getTypesCell]

theorem Bindings.function (path typesAddress numbersAddress prefixCell typesCell : Nat)
    (globals : Mutable.Globals) :
    Bindings (functionEnv
      ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).1
      ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2
      [.ref path, .ref typesAddress, .ref numbersAddress, .ref prefixCell, .ref typesCell] globals)
      path typesAddress numbersAddress prefixCell typesCell := by
  constructor <;> cases globals <;> rfl

theorem path_length_evaluate (view : PathView heap path nodes) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref path)) (globals : Mutable.Globals := []) :
    Evaluates program env condition heap (flag (decide (nodes.length = 6))) heap globals := by
  apply Evaluates.compare (middle := heap) (afterRight := heap) (finalHeap := heap)
    (av := number nodes.length) (bv := number 6)
  · apply Evaluates.builtin (.cons (Evaluates.lookup getPath) .nil)
    simp [primitive, Mutable.length, view.found, references, number]
  · exact Evaluates.constant
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, flag]
    omega

theorem set_membership_evaluate (heap : Mutable.Heap) (address : Nat) (items : List Int)
    (found : heap[address]? = some (integerSet items)) (setName itemName : String) (value : Int)
    (env : HeapCalls.Env) (getSet : env setName = some (.ref address))
    (getItem : env itemName = some (number value)) (globals : Mutable.Globals := []) :
    Evaluates program env (.membership (.name itemName) (.name setName) false) heap
      (flag (decide (value ∈ items))) heap globals := by
  apply Evaluates.membership (.cons (Evaluates.lookup getItem) (.cons (Evaluates.lookup getSet) .nil))
  simp [HeapCalls.membership, found, integerSet, ScalarSet.contains_ints, number, flag]

/-- The concrete return object represents the exact ordered result of the search model. -/
def ResultView (heap : Mutable.Heap) (result : Option ProjectEulerSolutions.PolygonalCycle.Path)
    (value : Mutable.Value) : Prop :=
  match result with
  | none => value = .scalar .none
  | some path => ∃ address nodes, value = .ref address ∧ Nonempty (PathView heap address nodes) ∧
      nodes.map Prod.snd = path

variable {result : Option ProjectEulerSolutions.PolygonalCycle.Path}

theorem ResultView.append (view : ResultView heap result value) (extra : Mutable.Heap) :
    ResultView (heap ++ extra) result value := by
  cases result with
  | none => exact view
  | some path =>
      obtain ⟨address, nodes, equal, ⟨pathView⟩, logical⟩ := view
      exact ⟨address, nodes, equal, ⟨pathView.append extra⟩, logical⟩

theorem result_condition_evaluate (view : ResultView heap result value) (env : HeapCalls.Env)
    (getResult : env "res" = some value) (globals : Mutable.Globals := []) :
    Evaluates program env resultCondition heap (flag result.isSome) heap globals := by
  apply Evaluates.unary (middle := heap) (finalHeap := heap)
    (value := flag (!result.isSome))
  · apply Evaluates.builtin (.cons (Evaluates.lookup getResult) (.cons Evaluates.constant .nil))
    cases result with
    | none => simp [ResultView] at view; subst value; rfl
    | some path => obtain ⟨address, _, rfl, _, _⟩ := view; rfl
  · cases result <;> rfl

def PathView.ends (view : PathView heap path nodes) (size : nodes.length = 6) :
    Ends heap path (nodes[0]'(by omega)).2.1 (nodes[5]'(by omega)).2.1 where
  items := references nodes
  size := by simp [references, size]
  pathFound := view.found
  firstAddress := (nodes[0]'(by omega)).1
  lastAddress := (nodes[5]'(by omega)).1
  firstKind := number (nodes[0]'(by omega)).2.2
  lastKind := number (nodes[5]'(by omega)).2.2
  head := by simp [references, List.getElem?_eq_getElem (show 0 < nodes.length by omega)]
  tail := by simp [references, List.getElem?_eq_getElem (show 5 < nodes.length by omega)]
  firstFound := view.contents _ _ (List.getElem_mem (by omega))
  lastFound := view.contents _ _ (List.getElem_mem (by omega))

/-- The source's six-node return is the model's closure test, including the failed case. -/
theorem finish_result (view : PathView heap path nodes) (size : nodes.length = 6)
    (usedTypes usedNums prefixCell typesCell : Mutable.Value) (globals : Mutable.Globals := []) :
    ∃ value, ReturnsValue program "__formalsw_local_solve_dfs"
      [.ref path, usedTypes, usedNums, prefixCell, typesCell] heap value heap globals ∧
      ResultView heap (ProjectEulerSolutions.PolygonalCycle.finish (nodes.map Prod.snd)) value := by
  let first := (nodes[0]'(by omega)).2.1
  let last := (nodes[5]'(by omega)).2.1
  have closed : ProjectEulerSolutions.PolygonalCycle.closed ((nodes.map Prod.snd).map Prod.fst) =
      decide (last % 100 = first / 100) := by
    simp [ProjectEulerSolutions.PolygonalCycle.closed, ProjectEulerSolutions.PolygonalCycle.links,
      List.getLast?_eq_getElem?, List.head?_eq_getElem?, size, first, last]
  refine ⟨if last % 100 = first / 100 then .ref path else .scalar .none,
    closed_result heap path first last (view.ends size) usedTypes usedNums prefixCell typesCell globals, ?_⟩
  by_cases closes : last % 100 = first / 100
  · simp only [ProjectEulerSolutions.PolygonalCycle.finish, closed, closes, decide_true, if_true]
    exact ⟨path, nodes, rfl, ⟨view⟩, rfl⟩
  · simp only [ProjectEulerSolutions.PolygonalCycle.finish, closed, closes, decide_false,
      Bool.false_eq_true, if_false]
    rfl

/-- A nonempty source path reads the same final number as the search model. -/
theorem path_last_evaluate (view : PathView heap path nodes) (target : Nat) (value kind : Int)
    (last : nodes.getLast? = some (target, (value, kind))) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref path)) (globals : Mutable.Globals := []) :
    Evaluates program env lastExpression heap (number value) heap globals := by
  have nonempty : nodes ≠ [] := by intro empty; simp [empty] at last
  have positive : 0 < nodes.length := List.length_pos_iff.mpr nonempty
  have tail : (references nodes)[nodes.length - 1]? = some (.ref target) := by
    simpa [references, List.getLast?_eq_getElem?] using
      congrArg (Option.map (fun node : Nat × Node => Mutable.Value.ref node.1)) last
  have offset : ((-1 : Int) + nodes.length).toNat = nodes.length - 1 := by omega
  have inner : Evaluates program env
      (.subscript (.name "path") (.unaryOp .uSub (.constant (.int 1)))) heap (.ref target) heap globals := by
    apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getPath)
      (Evaluates.unary (middle := heap) (finalHeap := heap) (value := number 1) Evaluates.constant rfl)
    have indexed : Mutable.index (references nodes) (number (-1)) heap =
        some (.ok (nodes.length - 1), heap) := by
      simp [Mutable.index, references, nonempty, offset]
    simp [Mutable.getItem, view.found, indexed, Except.map, tail]
  apply Evaluates.subscript (middle := heap) (afterIndex := heap) (finalHeap := heap)
    (by intros; intro impossible; cases impossible) inner Evaluates.constant
  have tuple := view.contents target (value, kind) (List.mem_of_getLast? last)
  simp [Mutable.getItem, tuple, Mutable.index, Except.map]

theorem suffix_evaluate (view : PathView heap path nodes) (target : Nat) (value kind : Int)
    (last : nodes.getLast? = some (target, (value, kind))) (env : HeapCalls.Env)
    (getPath : env "path" = some (.ref path)) (globals : Mutable.Globals := []) :
    Evaluates program env (.binOp lastExpression .mod (.constant (.int 100))) heap
      (number (value % 100)) heap globals := by
  apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
    (path_last_evaluate view target value kind last env getPath globals) Evaluates.constant
  simp [Mutable.binary, scalarBinary, number]

/-- A loop body resumes at its own boundary, including an explicit `continue`. -/
def IterationContinues (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap) (globals : Mutable.Globals := []) : Prop :=
  ∀ next following anchors stack, Reaches program anchors
    ⟨.block (body ++ [.loopResume next] ++ following) env, stack, heap⟩
    ⟨.block (next :: following) finalEnv, stack, finalHeap⟩ globals

theorem IterationContinues.nil (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    IterationContinues env [] heap env heap globals :=
  fun _ _ _ _ => .next rfl .refl

theorem IterationContinues.prepend
    (first : BlockContinues program env body heap middleEnv middleHeap globals)
    (rest : IterationContinues middleEnv tail middleHeap finalEnv finalHeap globals) :
    IterationContinues env (body ++ tail) heap finalEnv finalHeap globals := by
  intro next following anchors stack
  have joined :=
    (first (tail ++ [.loopResume next] ++ following) anchors stack).trans (rest next following anchors stack)
  simpa only [List.append_assoc] using joined

theorem IterationContinues.branch (flag : Bool)
    (evaluated : Evaluates program env test heap value middle globals)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : IterationContinues env ((if flag then yes else no) ++ body) middle finalEnv finalHeap globals) :
    IterationContinues env (.ifThen test yes no :: body) heap finalEnv finalHeap globals := by
  intro next following anchors stack
  have resumed : Reaches program anchors
      ⟨.block ((if flag then yes else no) ++ (body ++ [.loopResume next] ++ following)) env, stack, middle⟩
      ⟨.block (next :: following) finalEnv, stack, finalHeap⟩ globals := by
    have executed := rest next following anchors stack
    simpa only [List.append_assoc] using executed
  exact .next rfl ((evaluated anchors _).trans (.next (by simp [step, run, truth]) resumed))

theorem IterationContinues.skip (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (target : ∀ next following, Calls.loopTarget (body ++ [.loopResume next] ++ following) = some (next, following))
    (globals : Mutable.Globals := []) :
    IterationContinues env (.continueLoop :: body) heap env heap globals := by
  intro next following anchors stack
  have selected : Calls.loopTarget (body ++ .loopResume next :: following) = some (next, following) := by
    simpa using target next following
  exact .next (by simp [step, selected]) .refl

theorem recursive_evaluate (state : SearchState heap path typesAddress numbersAddress prefixCell typesCell nodes)
    (bound : Bindings env path typesAddress numbersAddress prefixCell typesCell) (value kind : Int)
    (freshKind : kind ∉ usedKinds nodes) (freshValue : value ∉ usedNumbers nodes)
    (getValue : env "nxt" = some (number value)) (getKind : env "t" = some (number kind))
    (returned : ReturnsValue program "__formalsw_local_solve_dfs"
      [.ref (heap.length + 2), .ref (heap.length + 4), .ref (heap.length + 6), .ref prefixCell, .ref typesCell]
      (argumentHeap heap nodes value kind) answer finalHeap globals) :
    Evaluates program env recursiveCall heap answer finalHeap globals :=
  Evaluates.call (arguments_evaluate state.pathView value kind typesAddress numbersAddress
    state.typesFound state.numbersFound freshKind freshValue env (.ref prefixCell) (.ref typesCell)
    bound.getPath bound.getTypes bound.getNumbers getValue getKind bound.getPrefixCell bound.getTypesCell globals)
    returned

theorem inner_skip (state : SearchState heap path typesAddress numbersAddress prefixCell typesCell nodes)
    (bound : Bindings env path typesAddress numbersAddress prefixCell typesCell) (value : Int)
    (used : value ∈ usedNumbers nodes) (getValue : env "nxt" = some (number value))
    (globals : Mutable.Globals := []) :
    IterationContinues env innerBody heap env heap globals := by
  have tested := set_membership_evaluate heap numbersAddress (usedNumbers nodes) state.numbersFound
    "used_nums" "nxt" value env bound.getNumbers getValue globals
  apply IterationContinues.branch (flag := true) (by simpa [used] using tested) rfl
  apply IterationContinues.skip
  intros
  rfl

theorem inner_continues (state : SearchState heap path typesAddress numbersAddress prefixCell typesCell nodes)
    (bound : Bindings env path typesAddress numbersAddress prefixCell typesCell) (value kind : Int)
    (freshKind : kind ∉ usedKinds nodes) (freshValue : value ∉ usedNumbers nodes)
    (getValue : env "nxt" = some (number value)) (getKind : env "t" = some (number kind))
    (returned : ReturnsValue program "__formalsw_local_solve_dfs"
      [.ref (heap.length + 2), .ref (heap.length + 4), .ref (heap.length + 6), .ref prefixCell, .ref typesCell]
      (argumentHeap heap nodes value kind) (.scalar .none) finalHeap globals) :
    IterationContinues env innerBody heap (setLocal env "res" (.scalar .none)) finalHeap globals := by
  have tested := set_membership_evaluate heap numbersAddress (usedNumbers nodes) state.numbersFound
    "used_nums" "nxt" value env bound.getNumbers getValue globals
  apply IterationContinues.branch (flag := false) (by simpa [freshValue] using tested) rfl
  apply IterationContinues.prepend (BlockContinues.assign
    (recursive_evaluate state bound value kind freshKind freshValue getValue getKind returned))
  apply IterationContinues.branch (flag := false)
    (result_condition_evaluate (result := none) (value := .scalar .none) rfl
      (setLocal env "res" (.scalar .none)) rfl globals) rfl
  exact IterationContinues.nil _ _ _

theorem inner_returns (state : SearchState heap path typesAddress numbersAddress prefixCell typesCell nodes)
    (bound : Bindings env path typesAddress numbersAddress prefixCell typesCell) (value kind : Int)
    (freshKind : kind ∉ usedKinds nodes) (freshValue : value ∉ usedNumbers nodes)
    (getValue : env "nxt" = some (number value)) (getKind : env "t" = some (number kind))
    (returned : ReturnsValue program "__formalsw_local_solve_dfs"
      [.ref (heap.length + 2), .ref (heap.length + 4), .ref (heap.length + 6), .ref prefixCell, .ref typesCell]
      (argumentHeap heap nodes value kind) answer finalHeap globals)
    (view : ResultView finalHeap (some logical) answer) (following : List Stmt) :
    BlockReturns program env (innerBody ++ following) heap answer finalHeap globals := by
  have tested := set_membership_evaluate heap numbersAddress (usedNumbers nodes) state.numbersFound
    "used_nums" "nxt" value env bound.getNumbers getValue globals
  apply BlockReturns.branch (flag := false) (by simpa [freshValue] using tested) rfl
  apply BlockReturns.assign (recursive_evaluate state bound value kind freshKind freshValue getValue getKind returned)
  apply BlockReturns.branch (flag := true)
    (result_condition_evaluate view (setLocal env "res" answer) rfl globals) rfl
  exact BlockReturns.return_value (Evaluates.lookup rfl)

/-- Exhausting a search block continues; finding a path returns before any following code. -/
def BlockOutcome (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap)
    (result : Option ProjectEulerSolutions.PolygonalCycle.Path) (answer : Mutable.Value)
    (globals : Mutable.Globals := []) : Prop :=
  match result with
  | none => BlockContinues program env body heap finalEnv finalHeap globals
  | some _ => ∀ following, BlockReturns program env (body ++ following) heap answer finalHeap globals

def IterationOutcome (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap)
    (result : Option ProjectEulerSolutions.PolygonalCycle.Path) (answer : Mutable.Value)
    (globals : Mutable.Globals := []) : Prop :=
  match result with
  | none => IterationContinues env body heap finalEnv finalHeap globals
  | some _ => ∀ following, BlockReturns program env (body ++ following) heap answer finalHeap globals

/-- Source iteration implements the model's ordered first-result operator, retaining heap effects. -/
theorem ordered_loop (all remaining : List Int) (position : Nat) (suffix : all.drop position = remaining)
    (source : Nat) (name : String) (body : List Stmt) (hasContinue : Calls.loopControl body = true)
    (visit : Int → Option ProjectEulerSolutions.PolygonalCycle.Path)
    (invariant : HeapCalls.Env → Mutable.Heap → Prop)
    (read : ∀ env heap, invariant env heap → heap[source]? = some (.list (numbers all)))
    (iterate : ∀ env heap value, invariant env heap → value ∈ all →
      ∃ extra finalEnv answer, invariant finalEnv (heap ++ extra) ∧
        ResultView (heap ++ extra) (visit value) answer ∧
        IterationOutcome (setLocal env name (number value)) body heap finalEnv (heap ++ extra)
          (visit value) answer globals)
    (env : HeapCalls.Env) (heap : Mutable.Heap) (good : invariant env heap) :
    ∃ extra finalEnv answer, invariant finalEnv (heap ++ extra) ∧
      ResultView (heap ++ extra) (ProjectEulerSolutions.PolygonalCycle.firstResult remaining visit) answer ∧
      BlockOutcome env [.sequenceStepLoop name (.ref source) position body []] heap finalEnv (heap ++ extra)
        (ProjectEulerSolutions.PolygonalCycle.firstResult remaining visit) answer globals := by
  induction remaining generalizing position env heap with
  | nil =>
      have sequence : Mutable.sequence (sequenceValue (.ref source)) heap = some (.ok (numbers all), heap) := by
        simp [sequenceValue, Mutable.sequence, read env heap good]
      have exhausted : all[position]? = none := by rw [← List.head?_drop, suffix]; rfl
      refine ⟨[], env, .scalar .none, by simpa using good, rfl, ?_⟩
      apply BlockContinues.sequence_done sequence
      · simp [numbers, List.getElem?_map, exhausted]
      · simpa using (BlockContinues.nil (module := program) (env := env) (heap := heap) (globals := globals))
  | cons value rest ih =>
      have selected : all[position]? = some value := by rw [← List.head?_drop, suffix]; rfl
      have member : value ∈ all := List.mem_of_getElem? selected
      have sequence : Mutable.sequence (sequenceValue (.ref source)) heap = some (.ok (numbers all), heap) := by
        simp [sequenceValue, Mutable.sequence, read env heap good]
      have item : (numbers all)[position]? = some (number value) := by
        simp [numbers, List.getElem?_map, selected]
      obtain ⟨extra, nextEnv, answer, nextGood, represented, executed⟩ := iterate env heap value good member
      cases outcome : visit value with
      | some logical =>
          refine ⟨extra, nextEnv, answer, nextGood, ?_, ?_⟩
          · simpa only [ProjectEulerSolutions.PolygonalCycle.firstResult, outcome] using represented
          · simp only [ProjectEulerSolutions.PolygonalCycle.firstResult, outcome, BlockOutcome]
            intro following
            apply BlockReturns.sequence_step sequence item
            have run := executed
            simp only [outcome, IterationOutcome] at run
            have returned := run
              ([Calls.loopContinuation body (.sequenceStepLoop name (.ref source) (position + 1) body [])] ++ following)
            simpa only [List.append_assoc, List.nil_append] using returned
      | none =>
          have nextSuffix : all.drop (position + 1) = rest := by
            simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]
          obtain ⟨later, finalEnv, finalAnswer, finalGood, finalView, finalRun⟩ :=
            ih (position + 1) nextSuffix nextEnv (heap ++ extra) nextGood
          refine ⟨extra ++ later, finalEnv, finalAnswer, ?_, ?_, ?_⟩
          · simpa only [List.append_assoc] using finalGood
          · simpa only [List.append_assoc, ProjectEulerSolutions.PolygonalCycle.firstResult, outcome] using finalView
          · simp only [ProjectEulerSolutions.PolygonalCycle.firstResult, outcome]
            have iteration : IterationContinues (setLocal env name (number value)) body heap nextEnv (heap ++ extra)
                globals := by simpa only [outcome, IterationOutcome] using executed
            cases result : ProjectEulerSolutions.PolygonalCycle.firstResult rest visit with
            | none =>
                simp only [BlockOutcome]
                have finished : BlockContinues program nextEnv
                    [.sequenceStepLoop name (.ref source) (position + 1) body []]
                    (heap ++ extra) finalEnv ((heap ++ extra) ++ later) globals := by
                  simpa only [result, BlockOutcome] using finalRun
                intro following anchors stack
                let next := Stmt.sequenceStepLoop name (.ref source) (position + 1) body []
                have traversal := (iteration next following anchors stack).trans (finished following anchors stack)
                have stepped : Reaches program anchors
                    ⟨.block (.sequenceStepLoop name (.ref source) position body [] :: following) env, stack, heap⟩
                    ⟨.block following finalEnv, stack, (heap ++ extra) ++ later⟩ globals :=
                  .next (by simp [step, run, sequence, item, Calls.loopContinuation, hasContinue, next]) traversal
                simpa only [List.append_assoc] using stepped
            | some logical =>
                simp only [BlockOutcome]
                have finished : ∀ following, BlockReturns program nextEnv
                    ([.sequenceStepLoop name (.ref source) (position + 1) body []] ++ following)
                    (heap ++ extra) finalAnswer ((heap ++ extra) ++ later) globals := by
                  simpa only [result, BlockOutcome] using finalRun
                intro following anchors stack
                let next := Stmt.sequenceStepLoop name (.ref source) (position + 1) body []
                have traversal := (iteration next following anchors stack).trans (finished following anchors stack)
                have stepped : Reaches program anchors
                    ⟨.block (.sequenceStepLoop name (.ref source) position body [] :: following) env, stack, heap⟩
                    ⟨.value finalAnswer, stack, (heap ++ extra) ++ later⟩ globals :=
                  .next (by simp [step, run, sequence, item, Calls.loopContinuation, hasContinue, next]) traversal
                simpa only [List.append_assoc] using stepped

/-- A candidate scan calls the real recursive function and skips exactly the already-used numbers. -/
theorem inner_loop (state : SearchState heap path typesAddress numbersAddress prefixCell typesCell nodes)
    (bound : Bindings env path typesAddress numbersAddress prefixCell typesCell)
    (kind suffix : Int) (freshKind : kind ∉ usedKinds nodes)
    (getKind : env "t" = some (number kind)) (getSuffix : env "suffix" = some (number suffix))
    (all : List Int) (source : Nat) (found : heap[source]? = some (.list (numbers all)))
    (visit : Int → Option ProjectEulerSolutions.PolygonalCycle.Path)
    (child : ∀ currentHeap value, SearchState currentHeap path typesAddress numbersAddress prefixCell typesCell nodes →
      value ∈ all → value ∉ usedNumbers nodes →
      ∃ extra answer, ReturnsValue program "__formalsw_local_solve_dfs"
        [.ref (currentHeap.length + 2), .ref (currentHeap.length + 4), .ref (currentHeap.length + 6),
         .ref prefixCell, .ref typesCell]
        (argumentHeap currentHeap nodes value kind) answer (argumentHeap currentHeap nodes value kind ++ extra) globals ∧
        ResultView (argumentHeap currentHeap nodes value kind ++ extra) (visit value) answer) :
    ∃ extra finalEnv answer, Bindings finalEnv path typesAddress numbersAddress prefixCell typesCell ∧
      finalEnv "suffix" = some (number suffix) ∧
      ResultView (heap ++ extra)
        (ProjectEulerSolutions.PolygonalCycle.firstResult all
          (fun value => if value ∈ usedNumbers nodes then none else visit value)) answer ∧
      BlockOutcome env [.sequenceStepLoop "nxt" (.ref source) 0 innerBody []] heap finalEnv (heap ++ extra)
        (ProjectEulerSolutions.PolygonalCycle.firstResult all
          (fun value => if value ∈ usedNumbers nodes then none else visit value)) answer globals := by
  let invariant := fun currentEnv currentHeap =>
    Nonempty (SearchState currentHeap path typesAddress numbersAddress prefixCell typesCell nodes) ∧
      Bindings currentEnv path typesAddress numbersAddress prefixCell typesCell ∧
      currentEnv "t" = some (number kind) ∧ currentEnv "suffix" = some (number suffix) ∧
      currentHeap[source]? = some (.list (numbers all))
  have iterate : ∀ currentEnv currentHeap value, invariant currentEnv currentHeap → value ∈ all →
      ∃ extra finalEnv answer, invariant finalEnv (currentHeap ++ extra) ∧
        ResultView (currentHeap ++ extra) (if value ∈ usedNumbers nodes then none else visit value) answer ∧
        IterationOutcome (setLocal currentEnv "nxt" (number value)) innerBody currentHeap finalEnv
          (currentHeap ++ extra) (if value ∈ usedNumbers nodes then none else visit value) answer globals := by
    intro currentEnv currentHeap value good member
    obtain ⟨⟨current⟩, currentBound, currentKind, currentSuffix, currentFound⟩ := good
    let chosenEnv := setLocal currentEnv "nxt" (number value)
    have chosenBound : Bindings chosenEnv path typesAddress numbersAddress prefixCell typesCell :=
      currentBound.local "nxt" (number value) (by decide)
    have chosenKind : chosenEnv "t" = some (number kind) := by simpa [chosenEnv, setLocal] using currentKind
    have chosenSuffix : chosenEnv "suffix" = some (number suffix) := by simpa [chosenEnv, setLocal] using currentSuffix
    by_cases used : value ∈ usedNumbers nodes
    · refine ⟨[], chosenEnv, .scalar .none, ?_, ?_, ?_⟩
      · exact ⟨⟨by simpa using current⟩, chosenBound, chosenKind, chosenSuffix, by simpa using currentFound⟩
      · simp only [if_pos used]; rfl
      · simp only [if_pos used, IterationOutcome]
        simpa only [List.append_nil] using inner_skip current chosenBound value used (by rfl) globals
    · obtain ⟨extra, answer, returned, represented⟩ := child currentHeap value current member used
      let allocations : Mutable.Heap :=
        [.tuple [number value, number kind], .list [.ref currentHeap.length],
         .list (references nodes ++ [.ref currentHeap.length]),
         .set [.int kind], integerSet (usedKinds nodes ++ [kind]),
         .set [.int value], integerSet (usedNumbers nodes ++ [value])]
      let total := allocations ++ extra
      have layout : argumentHeap currentHeap nodes value kind ++ extra = currentHeap ++ total := by
        simp [argumentHeap, pathHeap, total, allocations, List.append_assoc]
      rw [layout] at returned represented
      have finalGood : invariant (setLocal chosenEnv "res" answer) (currentHeap ++ total) := by
        refine ⟨⟨current.append total⟩, chosenBound.local "res" answer (by decide), ?_, ?_, ?_⟩
        · simpa [setLocal] using chosenKind
        · simpa [setLocal] using chosenSuffix
        · exact build_prefix_map.found_append _ _ _ _ currentFound
      refine ⟨total, setLocal chosenEnv "res" answer, answer, finalGood, ?_, ?_⟩
      · simpa only [if_neg used] using represented
      · simp only [if_neg used]
        cases outcome : visit value with
        | none =>
            have failed : answer = .scalar .none := by simpa only [outcome, ResultView] using represented
            subst answer
            exact inner_continues current chosenBound value kind freshKind used rfl chosenKind returned
        | some logical =>
            intro following
            exact inner_returns current chosenBound value kind freshKind used rfl chosenKind returned
              (by simpa only [outcome] using represented) following
  obtain ⟨extra, finalEnv, answer, good, represented, executed⟩ := ordered_loop all all 0 rfl source "nxt"
    innerBody (by rfl) (fun value => if value ∈ usedNumbers nodes then none else visit value) invariant
    (fun _ _ good => good.2.2.2.2) iterate env heap ⟨⟨state⟩, bound, getKind, getSuffix, found⟩
  exact ⟨extra, finalEnv, answer, good.2.1, good.2.2.2.1, represented, executed⟩

theorem BlockOutcome.for_sequence
    (ordinary : ∀ expressions, iterable ≠ .call (.name "range") expressions)
    (evaluated : Evaluates program env iterable heap value middle globals)
    (sequence : Mutable.sequence value middle = some (.ok items, middle))
    (rest : BlockOutcome env [.sequenceStepLoop name (sequenceSource value) 0 body []]
      middle finalEnv finalHeap result answer globals) :
    BlockOutcome env [.forIn (.name name) iterable body []] heap finalEnv finalHeap result answer globals := by
  cases result with
  | none => exact BlockContinues.for_sequence ordinary evaluated sequence rest
  | some logical =>
      intro following
      apply BlockReturns.for_sequence ordinary evaluated sequence
      exact rest following

theorem IterationOutcome.of_block
    (executed : BlockOutcome env body heap finalEnv finalHeap result answer globals) :
    IterationOutcome env body heap finalEnv finalHeap result answer globals := by
  cases result with
  | none =>
      have continued := IterationContinues.prepend executed (IterationContinues.nil finalEnv finalHeap globals)
      simpa only [List.append_nil] using continued
  | some logical => exact executed

theorem outer_skip (state : SearchState heap path typesAddress numbersAddress prefixCell typesCell nodes)
    (bound : Bindings env path typesAddress numbersAddress prefixCell typesCell) (kind : Int)
    (used : kind ∈ usedKinds nodes) (getKind : env "t" = some (number kind))
    (globals : Mutable.Globals := []) :
    IterationContinues env outerBody heap env heap globals := by
  have tested := set_membership_evaluate heap typesAddress (usedKinds nodes) state.typesFound
    "used_types" "t" kind env bound.getTypes getKind globals
  apply IterationContinues.branch (flag := true) (by simpa [used] using tested) rfl
  apply IterationContinues.skip
  intros
  rfl

def typeVisit (nodes : List (Nat × Node)) (suffix : Int)
    (visit : Int → Int → Option ProjectEulerSolutions.PolygonalCycle.Path) (kind : Int) :
    Option ProjectEulerSolutions.PolygonalCycle.Path :=
  if kind ∈ usedKinds nodes then none else
    ProjectEulerSolutions.PolygonalCycle.firstResult (build_prefix_map.bucket (candidates kind) suffix)
      (fun value => if value ∈ usedNumbers nodes then none else visit kind value)

/-- Both source loops retain their order, captured tables, and failed-call allocations. -/
theorem outer_loop (state : SearchState heap path typesAddress numbersAddress prefixCell typesCell nodes)
    (bound : Bindings env path typesAddress numbersAddress prefixCell typesCell)
    (suffix : Int) (getSuffix : env "suffix" = some (number suffix))
    (visit : Int → Int → Option ProjectEulerSolutions.PolygonalCycle.Path)
    (child : ∀ currentHeap kind value,
      SearchState currentHeap path typesAddress numbersAddress prefixCell typesCell nodes →
      kind ∈ solve_setup.types → kind ∉ usedKinds nodes →
      value ∈ build_prefix_map.bucket (candidates kind) suffix → value ∉ usedNumbers nodes →
      ∃ extra answer, ReturnsValue program "__formalsw_local_solve_dfs"
        [.ref (currentHeap.length + 2), .ref (currentHeap.length + 4), .ref (currentHeap.length + 6),
         .ref prefixCell, .ref typesCell]
        (argumentHeap currentHeap nodes value kind) answer (argumentHeap currentHeap nodes value kind ++ extra) globals ∧
        ResultView (argumentHeap currentHeap nodes value kind ++ extra) (visit kind value) answer) :
    ∃ extra finalEnv answer, Bindings finalEnv path typesAddress numbersAddress prefixCell typesCell ∧
      ResultView (heap ++ extra)
        (ProjectEulerSolutions.PolygonalCycle.firstResult solve_setup.types (typeVisit nodes suffix visit)) answer ∧
      BlockOutcome env [.forIn (.name "t") solve_setup.typeCell outerBody []] heap finalEnv (heap ++ extra)
        (ProjectEulerSolutions.PolygonalCycle.firstResult solve_setup.types (typeVisit nodes suffix visit)) answer globals := by
  let source := state.world.typesAddress
  let invariant := fun currentEnv currentHeap =>
    Nonempty (SearchState currentHeap path typesAddress numbersAddress prefixCell typesCell nodes) ∧
      Bindings currentEnv path typesAddress numbersAddress prefixCell typesCell ∧
      currentEnv "suffix" = some (number suffix) ∧
      currentHeap[source]? = some (.list (numbers solve_setup.types))
  have iterate : ∀ currentEnv currentHeap kind, invariant currentEnv currentHeap → kind ∈ solve_setup.types →
      ∃ extra finalEnv answer, invariant finalEnv (currentHeap ++ extra) ∧
        ResultView (currentHeap ++ extra) (typeVisit nodes suffix visit kind) answer ∧
        IterationOutcome (setLocal currentEnv "t" (number kind)) outerBody currentHeap finalEnv
          (currentHeap ++ extra) (typeVisit nodes suffix visit kind) answer globals := by
    intro currentEnv currentHeap kind good member
    obtain ⟨⟨current⟩, currentBound, currentSuffix, currentFound⟩ := good
    let chosenEnv := setLocal currentEnv "t" (number kind)
    have chosenBound : Bindings chosenEnv path typesAddress numbersAddress prefixCell typesCell :=
      currentBound.local "t" (number kind) (by decide)
    have chosenSuffix : chosenEnv "suffix" = some (number suffix) := by simpa [chosenEnv, setLocal] using currentSuffix
    by_cases used : kind ∈ usedKinds nodes
    · refine ⟨[], chosenEnv, .scalar .none, ?_, ?_, ?_⟩
      · exact ⟨⟨by simpa using current⟩, chosenBound, chosenSuffix, by simpa using currentFound⟩
      · simp only [typeVisit, if_pos used]; rfl
      · simp only [typeVisit, if_pos used, IterationOutcome]
        simpa only [List.append_nil] using outer_skip current chosenBound kind used rfl globals
    · obtain ⟨bucketAddress, evaluated, bucketFound⟩ := next_values_evaluate current.world kind suffix member chosenEnv
        chosenBound.getPrefixCell rfl chosenSuffix globals
      obtain ⟨extra, finalEnv, answer, finalBound, finalSuffix, represented, executed⟩ :=
        inner_loop (current.append [.list []]) chosenBound kind suffix used rfl chosenSuffix
          (build_prefix_map.bucket (candidates kind) suffix) bucketAddress bucketFound (visit kind)
          (fun nextHeap value ready inBucket fresh => child nextHeap kind value ready member used inBucket fresh)
      have sequence : Mutable.sequence (.ref bucketAddress) (currentHeap ++ [.list []]) =
          some (.ok (numbers (build_prefix_map.bucket (candidates kind) suffix)), currentHeap ++ [.list []]) := by
        simp [Mutable.sequence, bucketFound]
      have header := BlockOutcome.for_sequence
        (by intro expressions impossible; cases impossible) evaluated sequence executed
      let total : Mutable.Heap := .list [] :: extra
      have layout : (currentHeap ++ [.list []]) ++ extra = currentHeap ++ total := by
        simp [total, List.append_assoc]
      rw [layout] at represented header
      refine ⟨total, finalEnv, answer, ?_, ?_, ?_⟩
      · exact ⟨⟨current.append total⟩, finalBound, finalSuffix,
          build_prefix_map.found_append _ _ _ _ currentFound⟩
      · simpa only [typeVisit, if_neg used] using represented
      · simp only [typeVisit, if_neg used]
        have tested := set_membership_evaluate currentHeap typesAddress (usedKinds nodes) current.typesFound
          "used_types" "t" kind chosenEnv chosenBound.getTypes rfl globals
        cases outcome : ProjectEulerSolutions.PolygonalCycle.firstResult
            (build_prefix_map.bucket (candidates kind) suffix)
            (fun value => if value ∈ usedNumbers nodes then none else visit kind value) with
        | none =>
            apply IterationContinues.branch (flag := false) (by simpa [used] using tested) rfl
            have continued : IterationContinues chosenEnv [.forIn (.name "nxt") nextValues innerBody []]
                currentHeap finalEnv (currentHeap ++ total) globals :=
              IterationOutcome.of_block (result := none) (by simpa only [outcome] using header)
            exact continued
        | some logical =>
            intro following
            apply BlockReturns.branch (flag := false) (by simpa [used] using tested) rfl
            have returned : ∀ following, BlockReturns program chosenEnv
                ([.forIn (.name "nxt") nextValues innerBody []] ++ following) currentHeap answer
                (currentHeap ++ total) globals := by simpa only [outcome, BlockOutcome] using header
            exact returned following
  obtain ⟨extra, finalEnv, answer, good, represented, executed⟩ := ordered_loop solve_setup.types solve_setup.types 0 rfl
    source "t" outerBody rfl (typeVisit nodes suffix visit) invariant (fun _ _ good => good.2.2.2) iterate
    env heap ⟨⟨state⟩, bound, getSuffix, state.world.typesFound⟩
  refine ⟨extra, finalEnv, answer, good.2.1, represented, ?_⟩
  exact BlockOutcome.for_sequence (items := numbers solve_setup.types) (by intro expressions impossible; cases impossible)
    (types_evaluate state.world env bound.getTypesCell globals)
    (by simp [Mutable.sequence, state.world.typesFound]) executed

theorem BlockOutcome.complete
    (executed : BlockOutcome env body heap finalEnv finalHeap result answer globals)
    (represented : ResultView finalHeap result answer) :
    BlockReturns program env (body ++ [.return (.constant .none)]) heap answer finalHeap globals := by
  cases result with
  | none =>
      have failed : answer = .scalar .none := represented
      subst answer
      exact BlockContinues.returns executed (BlockReturns.return_value Evaluates.constant)
  | some logical => exact executed _

/-- Every reachable source DFS call terminates with the exact ordered search-model result. -/
theorem search_result (fuel : Nat) (heap : Mutable.Heap)
    (path typesAddress numbersAddress prefixCell typesCell : Nat) (nodes : List (Nat × Node))
    (state : SearchState heap path typesAddress numbersAddress prefixCell typesCell nodes)
    (nonempty : nodes ≠ []) (depth : nodes.length + fuel = 6) (globals : Mutable.Globals := []) :
    ∃ extra answer, ReturnsValue program "__formalsw_local_solve_dfs"
      [.ref path, .ref typesAddress, .ref numbersAddress, .ref prefixCell, .ref typesCell]
      heap answer (heap ++ extra) globals ∧
      ResultView (heap ++ extra) (ProjectEulerSolutions.PolygonalCycle.search fuel (nodes.map Prod.snd)) answer := by
  induction fuel generalizing heap path typesAddress numbersAddress prefixCell typesCell nodes with
  | zero =>
      have size : nodes.length = 6 := by omega
      obtain ⟨answer, returned, represented⟩ := finish_result state.pathView size (.ref typesAddress)
        (.ref numbersAddress) (.ref prefixCell) (.ref typesCell) globals
      refine ⟨[], answer, by simpa using returned, ?_⟩
      simpa [ProjectEulerSolutions.PolygonalCycle.search, size] using represented
  | succ fuel ih =>
      have short : nodes.length ≠ 6 := by omega
      obtain ⟨target, value, kind, last⟩ : ∃ target value kind, nodes.getLast? = some (target, (value, kind)) := by
        cases found : nodes.getLast? with
        | none => exact False.elim (nonempty (List.getLast?_eq_none_iff.mp found))
        | some item => exact ⟨item.1, item.2.1, item.2.2, rfl⟩
      let params := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).1
      let sourceBody := ((findFunctionIn program.body "__formalsw_local_solve_dfs").getD ([], [])).2
      let env := functionEnv params sourceBody
        [.ref path, .ref typesAddress, .ref numbersAddress, .ref prefixCell, .ref typesCell] globals
      have bound : Bindings env path typesAddress numbersAddress prefixCell typesCell :=
        Bindings.function path typesAddress numbersAddress prefixCell typesCell globals
      let suffixEnv := setLocal env "suffix" (number (value % 100))
      have suffixBound : Bindings suffixEnv path typesAddress numbersAddress prefixCell typesCell :=
        bound.local "suffix" (number (value % 100)) (by decide)
      let visit := fun nextKind nextValue =>
        ProjectEulerSolutions.PolygonalCycle.search fuel (nodes.map Prod.snd ++ [(nextValue, nextKind)])
      have child : ∀ currentHeap nextKind nextValue,
          SearchState currentHeap path typesAddress numbersAddress prefixCell typesCell nodes →
          nextKind ∈ solve_setup.types → nextKind ∉ usedKinds nodes →
          nextValue ∈ build_prefix_map.bucket (candidates nextKind) (value % 100) →
          nextValue ∉ usedNumbers nodes →
          ∃ extra answer, ReturnsValue program "__formalsw_local_solve_dfs"
            [.ref (currentHeap.length + 2), .ref (currentHeap.length + 4), .ref (currentHeap.length + 6),
             .ref prefixCell, .ref typesCell]
            (argumentHeap currentHeap nodes nextValue nextKind) answer
              (argumentHeap currentHeap nodes nextValue nextKind ++ extra) globals ∧
            ResultView (argumentHeap currentHeap nodes nextValue nextKind ++ extra) (visit nextKind nextValue) answer := by
        intro currentHeap nextKind nextValue ready _ _ _ _
        have childDepth : (nodes ++ [(currentHeap.length, (nextValue, nextKind))]).length + fuel = 6 := by
          simp only [List.length_append, List.length_singleton]
          omega
        have returned := ih (argumentHeap currentHeap nodes nextValue nextKind)
          (currentHeap.length + 2) (currentHeap.length + 4) (currentHeap.length + 6) prefixCell typesCell
          (nodes ++ [(currentHeap.length, (nextValue, nextKind))]) (ready.arguments nextValue nextKind)
          (by simp) childDepth
        simpa only [visit, List.map_append, List.map_cons, List.map_nil] using returned
      obtain ⟨extra, finalEnv, answer, finalBound, represented, executed⟩ :=
        outer_loop state suffixBound (value % 100) rfl visit child
      have model : ProjectEulerSolutions.PolygonalCycle.search (fuel + 1) (nodes.map Prod.snd) =
          ProjectEulerSolutions.PolygonalCycle.firstResult solve_setup.types (typeVisit nodes (value % 100) visit) := by
        rw [ProjectEulerSolutions.PolygonalCycle.search, List.length_map, if_neg short]
        simp only [List.getLast?_map, last, Option.map_some, Option.getD_some]
        simp only [List.map_map]
        rfl
      refine ⟨extra, answer, ?_, ?_⟩
      · apply BlockReturns.function rfl rfl
        change BlockReturns program env code heap answer (heap ++ extra) globals
        apply BlockReturns.branch (flag := false)
          (by simpa only [short, decide_false] using path_length_evaluate state.pathView env bound.getPath globals) rfl
        apply BlockReturns.assign (suffix_evaluate state.pathView target value kind last env bound.getPath globals)
        exact BlockOutcome.complete executed represented
      · rw [model]
        exact represented

def sumTarget : Expr := .tuple [.name "x", .name "_t"]
def sumExpression : Expr :=
  .sumGeneratorTargets (.name "x") [(sumTarget, .name "res", [])] (.constant (.int 0))
private def sumIteration : List Stmt := comprehensionTargetBody sumTarget (sumBody (.name "x") [])

private theorem sum_loop (view : PathView heap path nodes) (remaining : List (Nat × Node))
    (position : Nat) (suffix : nodes.drop position = remaining) (env : HeapCalls.Env) (initial : Int)
    (accumulator : env sumAccumulator = some (number initial)) (globals : Mutable.Globals) :
    BlockReturns program env
      [.sequenceStepLoop (comprehensionTargetName sumTarget) (.ref path) position sumIteration [],
       .return (.name sumAccumulator)] heap
      (number (remaining.foldl (fun total node => total + node.2.1) initial)) heap globals := by
  have sequence : Mutable.sequence (sequenceValue (.ref path)) heap = some (.ok (references nodes), heap) := by
    simp [sequenceValue, Mutable.sequence, view.found]
  induction remaining generalizing position env initial with
  | nil =>
      apply BlockReturns.sequence_done sequence
      · have exhausted : nodes[position]? = none := by rw [← List.head?_drop, suffix]; rfl
        simp [references, List.getElem?_map, exhausted]
      · exact BlockReturns.return_value (Evaluates.lookup accumulator)
  | cons head rest ih =>
      have selected : nodes[position]? = some head := by rw [← List.head?_drop, suffix]; rfl
      have tuple := view.contents head.1 head.2 (List.mem_of_getElem? selected)
      apply BlockReturns.sequence_step (value := .ref head.1) sequence
        (by simp [references, List.getElem?_map, selected])
      apply BlockReturns.unpack (middle := heap) (afterUnpack := heap)
        (names := ["x", "_t"]) (items := [number head.2.1, number head.2.2])
        rfl (Evaluates.lookup rfl) (by simp [Mutable.sequence, tuple]) rfl
      apply BlockReturns.assign (value := number (initial + head.2.1)) (middle := heap)
      · apply Evaluates.binary (middle := heap) (afterRight := heap) (finalHeap := heap)
          (av := number initial) (bv := number head.2.1)
        · apply Evaluates.lookup
          simpa [assignNames, setLocal, sumAccumulator, comprehensionTargetName, sumTarget] using accumulator
        · exact Evaluates.lookup (by simp [assignNames, setLocal])
        · simp [Mutable.binary, scalarBinary, number]
      · change BlockReturns program _
          [.sequenceStepLoop (comprehensionTargetName sumTarget) (.ref path) (position + 1) sumIteration [],
           .return (.name sumAccumulator)] heap _ heap globals
        apply ih
        · simp [List.drop_add_one_eq_tail_drop, suffix]
        · rfl

/-- The source generator unpacks the actual path tuples and sums their first components. -/
theorem sum_evaluate (view : PathView heap path nodes) (env : HeapCalls.Env)
    (getResult : env "res" = some (.ref path)) (globals : Mutable.Globals := []) :
    Evaluates program env sumExpression heap (number (usedNumbers nodes).sum) heap globals := by
  have total : nodes.foldl (fun total node => total + node.2.1) 0 = (usedNumbers nodes).sum := by
    simp [usedNumbers, List.sum_eq_foldl, List.foldl_map]
  apply Evaluates.sum_generator_targets_sequence (items := references nodes)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    (by intro expressions impossible; cases impossible)
    (Evaluates.lookup getResult) (by simp [Mutable.sequence, view.found]) Evaluates.constant rfl
  have executed := sum_loop view nodes 0 rfl
    (setLocal (comprehensionTargetsEnv env [(sumTarget, .name "res", [])]) sumAccumulator (number 0)) 0 rfl globals
  simpa only [total, comprehensionFilters, comprehensionTargetsBody, sumIteration] using executed

def startArguments : List Expr :=
  [.list [.tuple [.name "start", .constant (.int 8)]],
   .builtinCall .setDisplay [.constant (.int 8)], .builtinCall .setDisplay [.name "start"],
   .name "__formalsw_cell_solve_pref_map", .name "__formalsw_cell_solve_types"]
def startCall : Expr := .call (.name "__formalsw_local_solve_dfs") startArguments
def startHeap (heap : Mutable.Heap) (start : Int) : Mutable.Heap :=
  heap ++ [.tuple [number start, number 8], .list [.ref heap.length], .set [.int 8], .set [.int start]]

theorem start_arguments_evaluate (heap : Mutable.Heap) (start : Int) (env : HeapCalls.Env)
    (prefixCell typesCell : Nat) (getStart : env "start" = some (number start))
    (getPrefix : env "__formalsw_cell_solve_pref_map" = some (.ref prefixCell))
    (getTypes : env "__formalsw_cell_solve_types" = some (.ref typesCell)) (globals : Mutable.Globals := []) :
    Evaluates.Arguments program env startArguments heap
      [.ref (heap.length + 1), .ref (heap.length + 2), .ref (heap.length + 3), .ref prefixCell, .ref typesCell]
      (startHeap heap start) globals := by
  let pathReady : Mutable.Heap := heap ++ [.tuple [number start, number 8], .list [.ref heap.length]]
  let typesReady : Mutable.Heap := pathReady ++ [.set [.int 8]]
  have tuple : Evaluates program env (.tuple [.name "start", .constant (.int 8)]) heap (.ref heap.length)
      (heap ++ [.tuple [number start, number 8]]) globals :=
    Evaluates.tuple (.cons (Evaluates.lookup getStart) (.cons Evaluates.constant .nil)) (by simp)
  have pathRun : Evaluates program env (.list [.tuple [.name "start", .constant (.int 8)]]) heap
      (.ref (heap.length + 1)) pathReady globals := by
    have evaluated := Evaluates.list (.cons tuple .nil)
    simpa [pathReady, List.append_assoc] using evaluated
  have typesRun : Evaluates program env (.builtinCall .setDisplay [.constant (.int 8)]) pathReady
      (.ref (heap.length + 2)) typesReady globals := by
    apply Evaluates.builtin (.cons Evaluates.constant .nil)
    have allocated := Mutable.allocateSetValues_singleton pathReady (.int 8)
    simpa [pathReady, typesReady] using allocated
  have numbersRun : Evaluates program env (.builtinCall .setDisplay [.name "start"]) typesReady
      (.ref (heap.length + 3)) (startHeap heap start) globals := by
    apply Evaluates.builtin (.cons (Evaluates.lookup getStart) .nil)
    have allocated := Mutable.allocateSetValues_singleton typesReady (.int start)
    simpa [pathReady, typesReady, startHeap, List.append_assoc] using allocated
  exact .cons pathRun (.cons typesRun (.cons numbersRun
    (.cons (Evaluates.lookup getPrefix) (.cons (Evaluates.lookup getTypes) .nil))))

def startState (world : World heap prefixCell typesCell) (start : Int) :
    SearchState (startHeap heap start) (heap.length + 1) (heap.length + 2) (heap.length + 3)
      prefixCell typesCell [(heap.length, (start, 8))] where
  world := world.append [.tuple [number start, number 8], .list [.ref heap.length], .set [.int 8], .set [.int start]]
  pathView := {
    found := by simp [startHeap, references]
    contents := by
      intro target item member
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (List.mem_singleton.mp member)
      simp [startHeap]
  }
  typesFound := by simp [startHeap, integerSet, usedKinds]
  numbersFound := by simp [startHeap, integerSet, usedNumbers]

/-- Each real start call allocates its initial tuple, path, and two independent sets. -/
theorem start_evaluate (world : World heap prefixCell typesCell) (start : Int) (env : HeapCalls.Env)
    (getStart : env "start" = some (number start))
    (getPrefix : env "__formalsw_cell_solve_pref_map" = some (.ref prefixCell))
    (getTypes : env "__formalsw_cell_solve_types" = some (.ref typesCell)) (globals : Mutable.Globals := []) :
    ∃ extra answer, Evaluates program env startCall heap answer (heap ++ extra) globals ∧
      ResultView (heap ++ extra) (ProjectEulerSolutions.PolygonalCycle.search 5 [(start, 8)]) answer := by
  obtain ⟨extra, answer, returned, represented⟩ := search_result 5 (startHeap heap start)
    (heap.length + 1) (heap.length + 2) (heap.length + 3) prefixCell typesCell
    [(heap.length, (start, 8))] (startState world start) (by simp) (by simp) globals
  let total : Mutable.Heap :=
    [.tuple [number start, number 8], .list [.ref heap.length], .set [.int 8], .set [.int start]] ++ extra
  have layout : startHeap heap start ++ extra = heap ++ total := by simp [startHeap, total, List.append_assoc]
  refine ⟨total, answer, ?_, ?_⟩
  · rw [← layout]
    exact Evaluates.call (start_arguments_evaluate heap start env prefixCell typesCell getStart getPrefix getTypes globals)
      returned
  · rw [← layout]
    exact represented

def startBody : List Stmt :=
  [.assign [.name "res"] startCall, .ifThen resultCondition [.return sumExpression] []]
def startIterable : Expr := .subscript (.name "nums_by_type") (.constant (.int 8))

theorem start_source :
    (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 3 =
      .forIn (.name "start") startIterable startBody [] ::
        (executableBody ((findFunctionIn program.body "solve").getD ([], [])).2).drop 4 := rfl

theorem sum_result_evaluate (view : ResultView heap (some logical) answer) (env : HeapCalls.Env)
    (getResult : env "res" = some answer) (globals : Mutable.Globals := []) :
    Evaluates program env sumExpression heap (number (logical.map Prod.fst).sum) heap globals := by
  obtain ⟨path, nodes, rfl, ⟨pathView⟩, equal⟩ := view
  have numbers : logical.map Prod.fst = usedNumbers nodes := by
    rw [← equal]
    simp [usedNumbers, List.map_map]
  rw [numbers]
  exact sum_evaluate pathView env getResult globals

/-- A successful model start makes the source's terminal exception unreachable. -/
theorem start_loop (all remaining : List Int) (position : Nat) (suffix : all.drop position = remaining)
    (heap : Mutable.Heap) (prefixCell typesCell source : Nat) (world : World heap prefixCell typesCell)
    (found : heap[source]? = some (.list (numbers all))) (env : HeapCalls.Env)
    (getPrefix : env "__formalsw_cell_solve_pref_map" = some (.ref prefixCell))
    (getTypes : env "__formalsw_cell_solve_types" = some (.ref typesCell))
    (logical : ProjectEulerSolutions.PolygonalCycle.Path)
    (succeeds : ProjectEulerSolutions.PolygonalCycle.firstResult remaining
      (fun start => ProjectEulerSolutions.PolygonalCycle.search 5 [(start, 8)]) = some logical)
    (following : List Stmt) (globals : Mutable.Globals := []) :
    ∃ extra, BlockReturns program env
      (.sequenceStepLoop "start" (.ref source) position startBody [] :: following)
      heap (number (logical.map Prod.fst).sum) (heap ++ extra) globals := by
  induction remaining generalizing position heap env with
  | nil => simp [ProjectEulerSolutions.PolygonalCycle.firstResult] at succeeds
  | cons start rest ih =>
      have selected : all[position]? = some start := by rw [← List.head?_drop, suffix]; rfl
      have sequence : Mutable.sequence (sequenceValue (.ref source)) heap = some (.ok (numbers all), heap) := by
        simp [sequenceValue, Mutable.sequence, found]
      let bound := setLocal env "start" (number start)
      have boundPrefix : bound "__formalsw_cell_solve_pref_map" = some (.ref prefixCell) := by
        simpa [bound, setLocal] using getPrefix
      have boundTypes : bound "__formalsw_cell_solve_types" = some (.ref typesCell) := by
        simpa [bound, setLocal] using getTypes
      obtain ⟨extra, answer, evaluated, represented⟩ := start_evaluate world start bound rfl boundPrefix boundTypes globals
      let resultEnv := setLocal bound "res" answer
      have tested := result_condition_evaluate represented resultEnv rfl globals
      cases outcome : ProjectEulerSolutions.PolygonalCycle.search 5 [(start, 8)] with
      | none =>
          have tailSucceeds : ProjectEulerSolutions.PolygonalCycle.firstResult rest
              (fun start => ProjectEulerSolutions.PolygonalCycle.search 5 [(start, 8)]) = some logical := by
            simpa only [ProjectEulerSolutions.PolygonalCycle.firstResult, outcome] using succeeds
          have nextSuffix : all.drop (position + 1) = rest := by
            simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]
          obtain ⟨later, continued⟩ := ih (position + 1) nextSuffix (heap ++ extra) (world.append extra)
            (build_prefix_map.found_append _ _ _ _ found) resultEnv
            (by simpa [resultEnv, setLocal] using boundPrefix)
            (by simpa [resultEnv, setLocal] using boundTypes) tailSucceeds
          refine ⟨extra ++ later, ?_⟩
          apply BlockReturns.sequence_step (value := number start) sequence (by simp [numbers, List.getElem?_map, selected])
          apply BlockReturns.assign evaluated
          apply BlockReturns.branch (flag := false) (by simpa only [outcome] using tested) rfl
          simpa only [startBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            Bool.false_or, Bool.or_false, if_false, List.nil_append, List.append_assoc] using continued
      | some path =>
          have equal : path = logical := by
            simpa only [ProjectEulerSolutions.PolygonalCycle.firstResult, outcome, Option.some.injEq] using succeeds
          subst path
          refine ⟨extra, ?_⟩
          apply BlockReturns.sequence_step (value := number start) sequence (by simp [numbers, List.getElem?_map, selected])
          apply BlockReturns.assign evaluated
          apply BlockReturns.branch (flag := true) (by simpa only [outcome] using tested) rfl
          exact BlockReturns.return_value (sum_result_evaluate (by simpa only [outcome] using represented) resultEnv rfl globals)

end solvers.«61».dfs

namespace solvers.«61».solve
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

private abbrev program := Solvers.P61Ast.module
private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)

/-- The complete source solver follows the first successful start and preserves its caller's objects. -/
theorem model_result (caller : Mutable.Heap) (logical : ProjectEulerSolutions.PolygonalCycle.Path)
    (succeeds : ProjectEulerSolutions.PolygonalCycle.cycle = some logical) (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "solve" [] caller (number (logical.map Prod.fst).sum) (caller ++ extra) globals := by
  let params := ((findFunctionIn program.body "solve").getD ([], [])).1
  let sourceBody := ((findFunctionIn program.body "solve").getD ([], [])).2
  let env := functionEnv params sourceBody [] globals
  obtain ⟨preparedHeap, cell, ⟨prepared⟩, setup, cellFound⟩ := solve_setup.prepared_result caller env globals
  let readyEnv := setLocal (solve_setup.generatedEnv env caller) "__formalsw_cell_solve_pref_map" (.ref cell)
  let world := dfs.World.of_setup caller cell prepared cellFound
  have getPrefix : readyEnv "__formalsw_cell_solve_pref_map" = some (.ref cell) := rfl
  have getTypes : readyEnv "__formalsw_cell_solve_types" = some (.ref (caller.length + 1)) := by
    simp [readyEnv, solve_setup.generatedEnv, solve_setup.initialEnv, setLocal]
  have getNumbers : readyEnv "nums_by_type" = some (.ref (solve_setup.initialHeap caller).length) := by
    simp [readyEnv, solve_setup.generatedEnv, setLocal]
  obtain ⟨prefixExtra, prefixLayout⟩ := prepared.extension
  have table : preparedHeap[(solve_setup.initialHeap caller).length]? = some (.dict (solve_setup.numberEntries caller)) := by
    rw [prefixLayout]
    exact build_prefix_map.found_append _ _ _ _ (solve_setup.generated_table caller)
  have starts : preparedHeap[caller.length + 8]? =
      some (.list (dfs.numbers (ProjectEulerSolutions.PolygonalCycle.candidates 8))) := by
    rw [prefixLayout]
    have found := build_prefix_map.found_append _ prefixExtra _ _
      (solve_setup.generated_family caller 8 (by decide) (by decide))
    simpa only [solve_setup.numbers, dfs.numbers,
      generate_4digit_polygonals.candidates_eq_model 8 (by decide)] using found
  have iterable : Evaluates program readyEnv dfs.startIterable preparedHeap (.ref (caller.length + 8)) preparedHeap globals := by
    apply Evaluates.subscript (middle := preparedHeap) (afterIndex := preparedHeap) (finalHeap := preparedHeap)
      (by intros; intro impossible; cases impossible) (Evaluates.lookup getNumbers) Evaluates.constant
    exact Mutable.getItem_dictionary _ _ _ table (.int 8) (solve_setup.numberEntries_lookup caller 8 (by decide) (by decide))
  obtain ⟨searchExtra, searched⟩ := dfs.start_loop
    (ProjectEulerSolutions.PolygonalCycle.candidates 8) (ProjectEulerSolutions.PolygonalCycle.candidates 8) 0 rfl
    preparedHeap cell (caller.length + 1) (caller.length + 8) world starts readyEnv getPrefix getTypes logical succeeds
    ((executableBody sourceBody).drop 4) globals
  have tail : BlockReturns program readyEnv ((executableBody sourceBody).drop 3)
      preparedHeap (number (logical.map Prod.fst).sum) (preparedHeap ++ searchExtra) globals := by
    rw [dfs.start_source]
    apply BlockReturns.for_sequence (items := dfs.numbers (ProjectEulerSolutions.PolygonalCycle.candidates 8))
      (by intro expressions impossible; cases impossible) iterable (by simp [Mutable.sequence, starts])
    exact searched
  obtain ⟨generatedExtra, generatedLayout⟩ : ∃ extra, solve_setup.generatedHeap caller = caller ++ extra := by
    refine ⟨[.list (solve_setup.numbers solve_setup.types), .list [.ref caller.length]] ++
      (.dict (solve_setup.numberEntries caller) ::
        solve_setup.types.map (fun kind => .list (solve_setup.numbers (generate_4digit_polygonals.answer kind)))), ?_⟩
    rw [solve_setup.generated_layout]
    simp only [solve_setup.initialHeap, List.append_assoc]
  have finalLayout : preparedHeap ++ searchExtra = caller ++ (generatedExtra ++ prefixExtra ++ searchExtra) := by
    rw [prefixLayout, generatedLayout]
    simp only [List.append_assoc]
  refine ⟨generatedExtra ++ prefixExtra ++ searchExtra, ?_⟩
  rw [← finalLayout]
  apply BlockReturns.function rfl rfl
  have complete := BlockContinues.returns setup tail
  simpa only [List.take_append_drop] using complete

theorem heap_result (caller : Mutable.Heap) (globals : Mutable.Globals := []) :
    ∃ extra, ReturnsValue program "solve" [] caller (number 28684) (caller ++ extra) globals := by
  obtain ⟨extra, returned⟩ := model_result caller _ ProjectEulerSolutions.PolygonalCycle.cycle_checked globals
  exact ⟨extra, by simpa using returned⟩

theorem «Returns» : (SetCalls.call program "solve" [] []).Returns := by
  obtain ⟨extra, returned⟩ := heap_result []
  exact SetCalls.returns_of_heap returned.returns

theorem «Supported» : (SetCalls.call program "solve" [] []).Supported :=
  fun choices => («Returns» choices).supported

end solvers.«61».solve

namespace solvers.«61»

theorem solve_source_result (caller : FormalSW.PyAst.Mutable.Heap) (globals : FormalSW.PyAst.Mutable.Globals) :
    ∃ extra, FormalSW.PyAst.HeapCalls.ReturnsValue Solvers.P61Ast.module "solve" [] caller
      (.scalar (.int 28684)) (caller ++ extra) globals := solve.heap_result caller globals

end solvers.«61»
