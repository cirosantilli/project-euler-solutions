import «.formalsw».solvers.«18.py.ast»
import data.«project-euler-statements».data.py.project_euler_statements.«p18.py»
import FormalSW
import ProjectEulerStatements.Lib.TrianglePaths

namespace solvers.«18»
open FormalSW.PyAst
open ProjectEulerStatements.P18 (rowTotals reduceRowValues reducePrefix)
private abbrev program := Solvers.P18Ast.module
private abbrev encode (values : List Int) : List Mutable.Value :=
  values.map (fun value => .scalar (.int value))

private theorem prefix_read (heap extra : Mutable.Heap) (address : Nat) (object : Mutable.Object)
    (found : heap[address]? = some object) : (heap ++ extra)[address]? = some object := by
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).choose]
  exact found

private theorem index.result (heap : Mutable.Heap) (env : HeapCalls.Env) (name : String)
    (address position : Nat) (values : List Int) (index : Expr)
    (found : heap[address]? = some (.list (encode values))) (valid : position < values.length)
    (lookup : env name = some (.ref address))
    (evaluated : HeapCalls.Evaluates program env index heap (.scalar (.int position)) heap)
    (ordinary : ∀ a b c, index ≠ .slice a b c) :
    HeapCalls.Evaluates program env (.subscript (.name name) index) heap (.scalar (.int values[position])) heap := by
  apply HeapCalls.Evaluates.subscript ordinary
    (.lookup lookup) evaluated
  have read := Mutable.getItem_list heap address position (encode values) found (by simpa [encode] using valid)
  rw [List.getElem_map] at read
  exact read

private abbrev rowElement : Expr := .binOp (.subscript (.name "row") (.name "i")) .add
  (.builtinCall .max [.subscript (.name "dp") (.name "i"),
    .subscript (.name "dp") (.binOp (.name "i") .add (.constant (.int 1)))])

private theorem rowElement.result (heap : Mutable.Heap) (env : HeapCalls.Env)
    (rowAddress belowAddress position : Nat) (row below : List Int)
    (getRow : heap[rowAddress]? = some (.list (encode row)))
    (getBelow : heap[belowAddress]? = some (.list (encode below)))
    (inRow : position < row.length) (inNext : position + 1 < below.length)
    (localRow : env "row" = some (.ref rowAddress)) (localBelow : env "dp" = some (.ref belowAddress))
    (localIndex : env "i" = some (.scalar (.int position))) :
    HeapCalls.Evaluates program env rowElement heap
      (.scalar (.int (row[position] + max below[position] below[position + 1]))) heap := by
  have next : HeapCalls.Evaluates program env (.binOp (.name "i") .add (.constant (.int 1))) heap
      (.scalar (.int (position + 1))) heap := by
    simpa using HeapCalls.Evaluates.binary (op := .add) (.lookup localIndex)
      (HeapCalls.Evaluates.constant (value := .int 1)) (by rfl)
  apply HeapCalls.Evaluates.binary (bv := .scalar (.int (max below[position] below[position + 1]))) (afterRight := heap)
    (index.result heap env "row" rowAddress position row (.name "i") getRow inRow localRow (.lookup localIndex)
      (by intros a b c impossible; cases impossible)) ?_ rfl
  apply HeapCalls.Evaluates.builtin (.cons
    (index.result heap env "dp" belowAddress position below (.name "i") getBelow (by omega) localBelow (.lookup localIndex)
      (by intros a b c impossible; cases impossible))
    (.cons (index.result heap env "dp" belowAddress (position + 1) below _ getBelow inNext localBelow next (by intros a b c impossible; cases impossible)) .nil))
  by_cases larger : below[position] < below[position + 1]
  · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom, scalarCompare,
      evalCmpOp, HeapCalls.scalarResult, List.mapM, List.mapM.loop, larger, max_eq_right (le_of_lt larger)]
  · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom, scalarCompare,
      evalCmpOp, HeapCalls.scalarResult, List.mapM, List.mapM.loop, larger, max_eq_left (le_of_not_gt larger)]

private abbrev rowBody (address : Nat) : List Stmt := [.comprehensionAppend address rowElement]

private theorem rowLoop.result (heap : Mutable.Heap) (env : HeapCalls.Env)
    (rowAddress belowAddress position : Nat) (row below output : List Int)
    (getRow : heap[rowAddress]? = some (.list (encode row)))
    (getBelow : heap[belowAddress]? = some (.list (encode below))) (complete : row.length < below.length)
    (localRow : env "row" = some (.ref rowAddress)) (localBelow : env "dp" = some (.ref belowAddress)) :
    ∃ finalEnv, HeapCalls.BlockContinues program env
      [.rangeStepLoop "i" position row.length 1 (rowBody heap.length) []]
      (heap ++ [.list (encode output)]) finalEnv
      (heap ++ [.list (encode (output ++ (rowTotals row below).drop position))]) := by
  have count : (rowTotals row below).length = row.length := by
    rw [ProjectEulerStatements.P18.rowTotals_length, Nat.min_eq_left (by omega)]
  by_cases active : position < row.length
  · have nextValid : position + 1 < below.length := by omega
    let value := row[position] + max below[position] below[position + 1]
    let indexed := HeapCalls.setLocal env "i" (.scalar (.int position))
    have savedRow : indexed "row" = some (.ref rowAddress) := by simpa [indexed, HeapCalls.setLocal] using localRow
    have savedBelow : indexed "dp" = some (.ref belowAddress) := by simpa [indexed, HeapCalls.setLocal] using localBelow
    have emitted : HeapCalls.BlockContinues program indexed (rowBody heap.length)
        (heap ++ [.list (encode output)]) indexed (heap ++ [.list (encode (output ++ [value]))]) := by
      have step := HeapCalls.BlockContinues.comprehension_append (globals := [])
        (rowElement.result (heap ++ [.list (encode output)]) indexed rowAddress belowAddress position row below
          (prefix_read heap _ _ _ getRow) (prefix_read heap _ _ _ getBelow) active nextValid savedRow savedBelow rfl)
        (by simp : (heap ++ [.list (encode output)])[heap.length]? = some (.list (encode output)))
      simpa [rowBody, encode, value, List.map_append] using step
    obtain ⟨finalEnv, continued⟩ := rowLoop.result heap indexed rowAddress belowAddress (position + 1)
      row below (output ++ [value]) getRow getBelow complete savedRow savedBelow
    have inTotals : position < (rowTotals row below).length := by simpa [count] using active
    have total : (rowTotals row below)[position] = value := by
      have indexedValue := ProjectEulerStatements.P18.rowTotals_getElem? row below position
      simpa [List.getElem?_eq_getElem inTotals, List.getElem?_eq_getElem active,
        List.getElem?_eq_getElem (show position < below.length by omega),
        List.getElem?_eq_getElem nextValid, value] using indexedValue
    refine ⟨finalEnv, HeapCalls.BlockContinues.range_step (by simp [Calls.rangeContinues]; omega) ?_⟩
    simpa [rowBody, Calls.loopContinuation, Calls.loopControl, List.drop_eq_getElem_cons inTotals,
      total, List.append_assoc, Int.natCast_add] using emitted.append continued
  · refine ⟨env, ?_⟩
    have done : (rowTotals row below).drop position = [] := List.drop_eq_nil_iff.mpr (by omega)
    simpa [done] using HeapCalls.BlockContinues.range_done (module := program) (env := env)
      (name := "i") (next := (position : Int)) (stop := (row.length : Int)) (stride := 1)
      (body := rowBody heap.length) (otherwise := []) (heap := heap ++ [.list (encode output)])
      (by simp [Calls.rangeContinues]; omega) HeapCalls.BlockContinues.nil
termination_by row.length - position
decreasing_by omega

private abbrev rowComprehension : Expr := .listComp rowElement
  [("i", .call (.name "range") [.builtinCall .len [.name "row"]], [])]

private theorem rowComprehension.result (heap : Mutable.Heap) (env : HeapCalls.Env)
    (rowAddress belowAddress : Nat) (row below : List Int)
    (getRow : heap[rowAddress]? = some (.list (encode row)))
    (getBelow : heap[belowAddress]? = some (.list (encode below))) (complete : row.length < below.length)
    (localRow : env "row" = some (.ref rowAddress)) (localBelow : env "dp" = some (.ref belowAddress)) :
    HeapCalls.Evaluates program env rowComprehension heap (.ref heap.length)
      (heap ++ [.list (encode (rowTotals row below))]) := by
  let masked := HeapCalls.comprehensionEnv env
    [("i", .call (.name "range") [.builtinCall .len [.name "row"]], [])]
  obtain ⟨finalEnv, executed⟩ := rowLoop.result heap masked rowAddress belowAddress 0 row below []
    getRow getBelow complete
    (by simpa [masked, HeapCalls.comprehensionEnv] using localRow)
    (by simpa [masked, HeapCalls.comprehensionEnv] using localBelow)
  apply HeapCalls.Evaluates.list_comp_range (values := [.scalar (.int row.length)])
    (.cons (HeapCalls.Evaluates.builtin (.cons (.lookup localRow) .nil)
      (by simp [HeapCalls.primitive, Mutable.length, getRow, encode])) .nil) rfl
  simpa [rowBody, HeapCalls.comprehensionFilters, HeapCalls.comprehensionBody] using executed

private abbrev TriangleRows := List (Nat × List Int)
private def RowShape (rows : TriangleRows) : Prop :=
  ∀ (i : Nat) (valid : i < rows.length), rows[i].2.length = i + 1
private abbrev triangleBody : List Stmt :=
  [.assign [.name "row"] (.subscript (.name "triangle") (.name "r")),
    .assign [.name "dp"] rowComprehension]
private abbrev finalReturn : Stmt := .return (.subscript (.name "dp") (.constant (.int 0)))

private theorem triangleLoop.result (remaining : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (triangleAddress belowAddress : Nat) (rows : TriangleRows) (below : List Int)
    (getTriangle : heap[triangleAddress]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (shape : RowShape rows) (bounded : remaining ≤ rows.length)
    (getBelow : heap[belowAddress]? = some (.list (encode below))) (size : below.length = remaining + 1)
    (localTriangle : env "triangle" = some (.ref triangleAddress))
    (localBelow : env "dp" = some (.ref belowAddress)) :
    ∃ extra, HeapCalls.BlockReturns program env
      [.rangeStepLoop "r" ((remaining : Int) - 1) (-1) (-1) triangleBody [], finalReturn]
      heap (.scalar (.int ((reducePrefix (rows.map Prod.snd) remaining below).headD 0))) (heap ++ extra) := by
  induction remaining generalizing heap env belowAddress below with
  | zero =>
      refine ⟨[], ?_⟩
      apply HeapCalls.BlockReturns.range_done rfl
      apply HeapCalls.BlockReturns.return_value
      have head : below[0]'(by omega) = below.headD 0 := by
        cases below with
        | nil => simp at size
        | cons _ _ => rfl
      simpa [reducePrefix, head] using index.result heap env "dp" belowAddress 0 below (.constant (.int 0))
        getBelow (by omega) localBelow HeapCalls.Evaluates.constant
        (by intros a b c impossible; cases impossible)
  | succ remaining ih =>
      have valid : remaining < rows.length := by omega
      let pair := rows[remaining]
      have rowSize : pair.2.length = remaining + 1 := shape remaining valid
      let indexed := HeapCalls.setLocal env "r" (.scalar (.int remaining))
      let rowEnv := HeapCalls.setLocal indexed "row" (.ref pair.1)
      have readRow : HeapCalls.Evaluates program indexed (.subscript (.name "triangle") (.name "r"))
          heap (.ref pair.1) heap := by
        apply HeapCalls.Evaluates.subscript (by intros a b c impossible; cases impossible)
          (.lookup (by simpa [indexed, HeapCalls.setLocal] using localTriangle)) (.lookup rfl)
        have read := Mutable.getItem_list heap triangleAddress remaining
          (rows.map (fun pair => .ref pair.1)) getTriangle (by simpa using valid)
        rw [List.getElem_map] at read
        exact read
      have calculated := rowComprehension.result heap rowEnv pair.1 belowAddress pair.2 below
        (getRows pair (List.getElem_mem valid)) getBelow (by omega) rfl
        (by simpa [rowEnv, indexed, HeapCalls.setLocal] using localBelow)
      let newRow := rowTotals pair.2 below
      let extraRow := [Mutable.Object.list (encode newRow)]
      let assigned := HeapCalls.setLocal rowEnv "dp" (.ref heap.length)
      have newSize : newRow.length = remaining + 1 := by
        rw [ProjectEulerStatements.P18.rowTotals_length]
        simp [rowSize, size]
      obtain ⟨extra, continued⟩ := ih (heap ++ extraRow) assigned heap.length newRow
        (prefix_read heap extraRow _ _ getTriangle)
        (fun item member => prefix_read heap extraRow _ _ (getRows item member))
        (by omega) (by simp [extraRow]) newSize
        (by simpa [assigned, rowEnv, indexed, HeapCalls.setLocal] using localTriangle) rfl
      have same : reducePrefix (rows.map Prod.snd) (remaining + 1) below =
          reducePrefix (rows.map Prod.snd) remaining newRow := by
        rw [ProjectEulerStatements.P18.reducePrefix_succ _ remaining below (by simpa using valid)]
        simp only [List.getElem_map]
        rfl
      refine ⟨extraRow ++ extra, ?_⟩
      rw [show ((remaining + 1 : Nat) : Int) - 1 = (remaining : Int) by omega, same]
      apply HeapCalls.BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply HeapCalls.BlockReturns.assign readRow
      apply HeapCalls.BlockReturns.assign calculated
      simpa [triangleBody, Calls.loopContinuation, Calls.loopControl,
        List.append_assoc, sub_eq_add_neg] using continued

private abbrev lastExpression : Expr :=
  .subscript (.name "triangle") (.unaryOp .uSub (.constant (.int 1)))

private theorem max_path_sum.result (heap : Mutable.Heap) (address : Nat) (rows : TriangleRows)
    (getTriangle : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (shape : RowShape rows) (nonempty : rows ≠ []) :
    ∃ extra, HeapCalls.ReturnsValue program "max_path_sum" [.ref address] heap
      (.scalar (.int ((reduceRowValues (rows.map Prod.snd)).headD 0))) (heap ++ extra) := by
  let env := HeapCalls.argumentEnv
    [⟨"triangle", some (.subscript (.name "list") (.subscript (.name "list") (.name "int"))), none⟩]
    [.ref address]
  have positive : 0 < rows.length := List.length_pos_iff.mpr nonempty
  have valid : rows.length - 1 < rows.length := by omega
  let bottom := rows[rows.length - 1]
  let extraBottom := [Mutable.Object.list (encode bottom.2)]
  let copiedHeap := heap ++ extraBottom
  let dpEnv := HeapCalls.setLocal env "dp" (.ref heap.length)
  have shifted : (-1 + (rows.length : Int)).toNat = rows.length - 1 := by omega
  have indexed : Mutable.index (rows.map (fun pair => Mutable.Value.ref pair.1)) (.scalar (.int (-1))) heap =
      some (.ok (rows.length - 1), heap) := by
    simp [Mutable.index, nonempty, shifted]
  have last : HeapCalls.Evaluates program env lastExpression heap (.ref bottom.1) heap := by
    apply HeapCalls.Evaluates.subscript (by intros a b c impossible; cases impossible) (.lookup rfl)
      (HeapCalls.Evaluates.unary HeapCalls.Evaluates.constant rfl)
    simp [Mutable.getItem, getTriangle, indexed, Except.map, List.getElem?_eq_getElem valid, bottom]
  have copied : HeapCalls.Evaluates program env (.subscript lastExpression (.slice none none none)) heap
      (.ref heap.length) copiedHeap :=
    HeapCalls.Evaluates.slice (.cons last (.cons .constant (.cons .constant (.cons .constant .nil))))
      (Mutable.getSlice_list_copy heap bottom.1 (encode bottom.2) (getRows bottom (List.getElem_mem valid)))
  obtain ⟨extra, loop⟩ := triangleLoop.result (rows.length - 1) copiedHeap dpEnv address heap.length rows bottom.2
    (prefix_read heap extraBottom _ _ getTriangle)
    (fun pair member => prefix_read heap extraBottom _ _ (getRows pair member)) shape (by omega)
    (by simp [copiedHeap, extraBottom]) (shape _ valid) rfl rfl
  have same : reduceRowValues (rows.map Prod.snd) =
      reducePrefix (rows.map Prod.snd) (rows.length - 1) bottom.2 := by
    simpa only [List.length_map, List.getElem_map] using
      ProjectEulerStatements.P18.reduceRowValues_from_last (rows.map Prod.snd) (by simpa using nonempty)
  refine ⟨extraBottom ++ extra, ?_⟩
  apply HeapCalls.BlockReturns.function rfl rfl
  apply HeapCalls.BlockReturns.assign copied
  apply HeapCalls.BlockReturns.for_range (start := ((rows.length - 1 : Nat) : Int) - 1) (stop := -1) (stride := -1)
    (values := [.scalar (.int ((rows.length : Int) - 2)), .scalar (.int (-1)), .scalar (.int (-1))])
  · apply HeapCalls.Evaluates.Arguments.cons
    · apply HeapCalls.Evaluates.binary (av := .scalar (.int rows.length)) (middle := copiedHeap)
      · exact HeapCalls.Evaluates.builtin (.cons (.lookup rfl) .nil)
          (by simp [HeapCalls.primitive, Mutable.length, copiedHeap, prefix_read heap extraBottom _ _ getTriangle])
      · exact .constant
      · rfl
    · exact .cons (HeapCalls.Evaluates.unary .constant rfl) (.cons (HeapCalls.Evaluates.unary .constant rfl) .nil)
  · have start : (rows.length : Int) - 2 = ((rows.length - 1 : Nat) : Int) - 1 := by omega
    simpa only [start] using (show HeapCalls.rangeBounds
      [.scalar (.int (((rows.length - 1 : Nat) : Int) - 1)), .scalar (.int (-1)), .scalar (.int (-1))] =
      .ok (((rows.length - 1 : Nat) : Int) - 1, -1, -1) from rfl)
  · simpa only [same, copiedHeap, List.append_assoc] using loop

private abbrev lengthTest : Expr :=
  .compare (.builtinCall .len [.name "row"]) [(.notEq, .name "size")]
private abbrev shapeBody : List Stmt :=
  [.ifThen lengthTest [.return (.constant (.bool false))] [],
    .augAssign (.name "size") .add (.constant (.int 1))]

private theorem rows_check (heap : Mutable.Heap) (address : Nat) (rows : TriangleRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)))
    (position : Nat) (env : HeapCalls.Env) (localSize : env "size" = some (.scalar (.int (position + 1)))) :
    ((∀ (i : Nat) (valid : i < rows.length), position ≤ i → rows[i].2.length = i + 1) ∧
      ∃ finalEnv, HeapCalls.BlockContinues program env
        [.sequenceStepLoop "row" (.ref address) position shapeBody []] heap finalEnv heap) ∨
    (∀ following, HeapCalls.BlockReturns program env
      (.sequenceStepLoop "row" (.ref address) position shapeBody [] :: following)
      heap (.scalar (.bool false)) heap) := by
  have sequence : Mutable.sequence (.ref address) heap =
      some (.ok (rows.map (fun pair => .ref pair.1)), heap) := by simp [Mutable.sequence, found]
  by_cases active : position < rows.length
  · let pair := rows[position]
    let boundEnv := HeapCalls.setLocal env "row" (.ref pair.1)
    have item : (rows.map (fun pair => Mutable.Value.ref pair.1))[position]? = some (.ref pair.1) := by
      simp [List.getElem?_eq_getElem active, pair]
    have savedSize : boundEnv "size" = some (.scalar (.int (position + 1))) := by
      simpa [boundEnv, HeapCalls.setLocal] using localSize
    have tested : HeapCalls.Evaluates program boundEnv lengthTest heap
        (.scalar (.bool (decide (pair.2.length ≠ position + 1)))) heap := by
      apply HeapCalls.Evaluates.compare (av := .scalar (.int pair.2.length)) (middle := heap)
      · exact HeapCalls.Evaluates.builtin (.cons (.lookup rfl) .nil)
          (by simp [HeapCalls.primitive, Mutable.length, reads pair (List.getElem_mem active), encode])
      · exact .lookup savedSize
      · simp [HeapCalls.compare, scalarCompare, evalCmpOp]
        omega
    by_cases good : pair.2.length = position + 1
    · let nextEnv := HeapCalls.setLocal boundEnv "size" (.scalar (.int (position + 1 + 1)))
      have step : HeapCalls.BlockContinues program boundEnv shapeBody heap nextEnv heap := by
        have skip : HeapCalls.BlockContinues program boundEnv
            [.ifThen lengthTest [.return (.constant (.bool false))] []] heap boundEnv heap :=
          HeapCalls.BlockContinues.branch (flag := false) (by simpa [good] using tested) rfl HeapCalls.BlockContinues.nil
        apply skip.append
        exact HeapCalls.BlockContinues.augAssign savedSize HeapCalls.Evaluates.constant
          (by simp [HeapCalls.inplace, Mutable.binary, scalarBinary])
      rcases rows_check heap address rows found reads (position + 1) nextEnv rfl with
        ⟨goodRows, finalEnv, continued⟩ | rejected
      · left
        refine ⟨?_, finalEnv, ?_⟩
        · intro i valid after
          by_cases same : i = position
          · subst i
            exact good
          · exact goodRows i valid (by omega)
        · apply HeapCalls.BlockContinues.sequence_step (source := .ref address) sequence item
          simpa [shapeBody, Calls.loopContinuation, Calls.loopControl] using step.append continued
      · right
        intro following
        apply HeapCalls.BlockReturns.sequence_step (source := .ref address) sequence item
        simpa [shapeBody, Calls.loopContinuation, Calls.loopControl] using step.returns (rejected following)
    · right
      intro following
      apply HeapCalls.BlockReturns.sequence_step (source := .ref address) sequence item
      exact HeapCalls.BlockReturns.branch (flag := true) (by simpa [good] using tested) rfl
        (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
  · left
    refine ⟨by intros; omega, env, ?_⟩
    exact HeapCalls.BlockContinues.sequence_done (source := .ref address) sequence
      (by simp [List.getElem?_eq_none_iff.mpr (Nat.le_of_not_gt active)]) HeapCalls.BlockContinues.nil
termination_by rows.length - position
decreasing_by omega

private theorem is_triangle.result (heap : Mutable.Heap) (address : Nat) (rows : TriangleRows)
    (found : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) :
    (rows ≠ [] ∧ RowShape rows ∧
      HeapCalls.ReturnsValue program "is_triangle" [.ref address] heap (.scalar (.bool true)) heap) ∨
      HeapCalls.ReturnsValue program "is_triangle" [.ref address] heap (.scalar (.bool false)) heap := by
  let args : List Arg := [⟨"triangle", some (.subscript (.name "list") (.subscript (.name "list") (.name "int"))), none⟩]
  let env := HeapCalls.argumentEnv args [.ref address]
  have tested : HeapCalls.Evaluates program env (.unaryOp .not (.name "triangle")) heap
      (.scalar (.bool (decide (rows = [])))) heap :=
    HeapCalls.Evaluates.unary (.lookup rfl)
      (by simp [HeapCalls.unary, Mutable.truth, found]; cases rows <;> rfl)
  by_cases empty : rows = []
  · right
    apply HeapCalls.BlockReturns.function (args := args) rfl rfl
    exact HeapCalls.BlockReturns.branch (flag := true) (by simpa [empty] using tested) rfl
      (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
  · let sized := HeapCalls.setLocal env "size" (.scalar (.int 1))
    have sequence : Mutable.sequence (.ref address) heap =
        some (.ok (rows.map (fun pair => .ref pair.1)), heap) := by simp [Mutable.sequence, found]
    rcases rows_check heap address rows found reads 0 sized rfl with ⟨good, finalEnv, loop⟩ | rejected
    · left
      refine ⟨empty, fun i valid => good i valid (by omega), ?_⟩
      apply HeapCalls.BlockReturns.function (args := args) rfl rfl
      apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [empty] using tested) rfl
      apply HeapCalls.BlockReturns.assign (name := "size") HeapCalls.Evaluates.constant
      have entered : HeapCalls.BlockContinues program sized
          [.forIn (.name "row") (.name "triangle") shapeBody []] heap finalEnv heap :=
        HeapCalls.BlockContinues.for_sequence (by intros expressions impossible; cases impossible)
          (.lookup rfl) sequence loop
      exact entered.returns (HeapCalls.BlockReturns.return_value HeapCalls.Evaluates.constant)
    · right
      apply HeapCalls.BlockReturns.function (args := args) rfl rfl
      apply HeapCalls.BlockReturns.branch (flag := false) (by simpa [empty] using tested) rfl
      apply HeapCalls.BlockReturns.assign (name := "size") HeapCalls.Evaluates.constant
      exact HeapCalls.BlockReturns.for_sequence (by intros expressions impossible; cases impossible)
        (.lookup rfl) sequence (rejected [.return (.constant (.bool true))])

private def TriangleDomain (heap : Mutable.Heap) (address : Nat) : Prop :=
  ∃ rows : TriangleRows,
    heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))) ∧
    (∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) ∧ RowShape rows ∧ rows ≠ []

private theorem decode_rows (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    ∃ rows : TriangleRows,
      heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))) ∧
      ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2)) := by
  classical
  obtain ⟨addresses, found, rowTypes⟩ := Mutable.Value.lists_of_hasType typed
  have decoded : ∀ row ∈ addresses, ∃ values : List Int, heap[row]? = some (.list (encode values)) :=
    fun row member => Mutable.Value.ints_of_hasType (rowTypes row member)
  choose values contents using decoded
  let rows := addresses.attach.map (fun entry => (entry.val, values entry.val entry.property))
  have same : rows.map (fun pair => Mutable.Value.ref pair.1) = addresses.map Mutable.Value.ref := by
    simp [rows, List.map_map]
  refine ⟨rows, by simpa [same] using found, ?_⟩
  intro pair member
  obtain ⟨entry, _, rfl⟩ := List.mem_map.mp member
  exact contents entry.val entry.property

private abbrev triangleEnv (address : Nat) : HeapCalls.Env :=
  fun name => if name = "triangle" then some (.ref address) else none

private theorem triangle_condition (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    ∃ flag, HeapCalls.Evaluates program (triangleEnv address) Solvers.P18Ast.requires_max_path_sum_0 heap
      (.scalar (.bool flag)) heap ∧ (flag = true → TriangleDomain heap address) := by
  obtain ⟨rows, found, reads⟩ := decode_rows heap address typed
  rcases is_triangle.result heap address rows found reads with ⟨nonempty, shape, executed⟩ | executed
  · exact ⟨true, HeapCalls.Evaluates.call (.cons (.lookup rfl) .nil) executed,
      fun _ => ⟨rows, found, reads, shape, nonempty⟩⟩
  · exact ⟨false, HeapCalls.Evaluates.call (.cons (.lookup rfl) .nil) executed, by simp⟩

private theorem triangle_domain_of_condition (heap : Mutable.Heap) (address : Nat)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (requires : HeapCalls.Condition program (triangleEnv address) Solvers.P18Ast.requires_max_path_sum_0 heap) :
    TriangleDomain heap address := by
  obtain ⟨flag, evaluated, domain⟩ := triangle_condition heap address typed
  exact domain (by simpa using evaluated.condition_iff.mp requires)

theorem requires_max_path_sum_0_supported (heap : Mutable.Heap) (address : Nat) (_valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]") :
    (HeapCalls.evaluate program (triangleEnv address) Solvers.P18Ast.requires_max_path_sum_0 heap).Supported := by
  obtain ⟨_, evaluated, _⟩ := triangle_condition heap address typed
  exact Execution.supported_of_finishes evaluated.finishes

theorem max_path_sum.«Equiv(naive)» (heap : Mutable.Heap) (address : Nat) (valid : heap.Valid)
    (typed : Mutable.Value.HasType heap (.ref address) "list[list[int]]")
    (requires : HeapCalls.Condition program (triangleEnv address) Solvers.P18Ast.requires_max_path_sum_0 heap) :
    (HeapCalls.call program "max_path_sum" [.ref address] heap).Equiv
      (HeapCalls.call P18Ast.module "naive" [.ref address] heap) := by
  obtain ⟨rows, found, reads, shape, nonempty⟩ := triangle_domain_of_condition heap address typed requires
  obtain ⟨extra, optimized⟩ := max_path_sum.result heap address rows found reads shape nonempty
  obtain ⟨referenceExtra, reference⟩ :=
    data.«project-euler-statements».data.py.project_euler_statements.p18.naive.result heap address rows found reads
  exact Mutable.Call.equiv_of_finishes optimized.finishes reference.finishes
    (Mutable.Completion.equivalent_allocations heap extra referenceExtra
      (.scalar (.int ((reduceRowValues (rows.map Prod.snd)).headD 0))) [] valid trivial (by simp))

theorem max_path_sum.«Supported» :
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[list[int]]" →
      HeapCalls.Condition program (triangleEnv address) Solvers.P18Ast.requires_max_path_sum_0 heap →
      (HeapCalls.call program "max_path_sum" [.ref address] heap).Supported) ∧
    (∀ (heap : Mutable.Heap) (address : Nat), heap.Valid →
      Mutable.Value.HasType heap (.ref address) "list[list[int]]" →
      HeapCalls.Condition program (triangleEnv address) Solvers.P18Ast.requires_max_path_sum_0 heap →
      (HeapCalls.call P18Ast.module "naive" [.ref address] heap).Supported) := by
  constructor
  · intro heap address _ typed requires
    obtain ⟨rows, found, reads, shape, nonempty⟩ := triangle_domain_of_condition heap address typed requires
    obtain ⟨_, executed⟩ := max_path_sum.result heap address rows found reads shape nonempty
    exact executed.returns.supported
  · intro heap address valid typed _
    exact data.«project-euler-statements».data.py.project_euler_statements.p18.naive.«Supported» heap address valid typed

end solvers.«18»
