import «.formalsw».solvers.«61.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p61.py»
import FormalSW
import ProjectEulerSolutions.Lib.PolygonalCycle

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

namespace solvers.«61»

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
