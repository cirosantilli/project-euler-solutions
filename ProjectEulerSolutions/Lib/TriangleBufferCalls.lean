import lang.py.FormalSW.HeapCalls
import ProjectEulerStatements.Lib.TriangleBuffers

/-! Source execution of ascending adjacent-child updates in a private copied row. -/
namespace ProjectEulerSolutions.TriangleBufferCalls
open FormalSW.PyAst HeapCalls ProjectEulerStatements.P18
set_option Elab.async false

abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
def encode (values : List Int) : List Mutable.Value := values.map number
def working (base : Mutable.Heap) (values : List Int) : Mutable.Heap := base ++ [.list (encode values)]

theorem found (base : Mutable.Heap) (values : List Int) :
    (working base values)[base.length]? = some (.list (encode values)) := by simp [working]

theorem prefix_found (base : Mutable.Heap) (values : List Int) (address : Nat) (object : Mutable.Object)
    (read : base[address]? = some object) : (working base values)[address]? = some object := by
  unfold working
  rw [List.getElem?_append_left (List.getElem?_eq_some_iff.mp read).1]
  exact read

theorem index_evaluates (program : Module) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (name : String) (address index : Nat) (items : List Int) (value : Int) (key : Expr)
    (receiver : env name = some (.ref address))
    (evaluated : Evaluates program env key heap (number index) heap)
    (ordinary : ∀ a b c, key ≠ .slice a b c)
    (contents : heap[address]? = some (.list (encode items))) (read : items[index]? = some value) :
    Evaluates program env (.subscript (.name name) key) heap (number value) heap := by
  obtain ⟨valid, same⟩ := List.getElem?_eq_some_iff.mp read
  apply Evaluates.subscript ordinary (Evaluates.lookup receiver) evaluated
  have got := Mutable.getItem_list heap address index (encode items) contents (by simpa [encode] using valid)
  simpa only [encode, List.getElem_map, same] using got

theorem set_item (base : Mutable.Heap) (items : List Int) (index : Nat) (value : Int)
    (inside : index < items.length) :
    Mutable.setItem (.ref base.length) (number index) (number value) (working base items) =
      some (.ok (.scalar .none), working base (items.set index value)) := by
  have nonnegative : ¬ (index : Int) < 0 := by omega
  have bounded : ¬ items.length ≤ index := by omega
  simp [Mutable.setItem, Mutable.index, working, encode, number, nonnegative, bounded,
    Except.map, List.set_append_right, List.map_set]

def element : Expr := .binOp (.subscript (.name "row") (.name "j")) .add
  (.builtinCall .max [.subscript (.name "dp") (.name "j"),
    .subscript (.name "dp") (.binOp (.name "j") .add (.constant (.int 1)))])
def body : List Stmt := [.assign [.subscript (.name "dp") (.name "j")] element]

theorem element_result (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (rowAddress cursor : Nat) (row below : List Int)
    (getRow : base[rowAddress]? = some (.list (encode row)))
    (complete : row.length < below.length) (inside : cursor < row.length)
    (localRow : env "row" = some (.ref rowAddress)) (localBelow : env "dp" = some (.ref base.length))
    (localCursor : env "j" = some (number cursor)) :
    Evaluates program env element (working base (partialRow row below cursor))
      (number (row[cursor] + max below[cursor] below[cursor + 1]))
      (working base (partialRow row below cursor)) := by
  have nextIndex : Evaluates program env (.binOp (.name "j") .add (.constant (.int 1)))
      (working base (partialRow row below cursor)) (number (cursor + 1 : Nat))
      (working base (partialRow row below cursor)) := by
    exact Evaluates.binary (Evaluates.lookup localCursor) Evaluates.constant
      (by simp [Mutable.binary, scalarBinary, number])
  have getCurrent : (partialRow row below cursor)[cursor]? = some below[cursor] := by
    rw [partialRow_suffix row below cursor cursor complete inside.le (by omega)]
    exact List.getElem?_eq_getElem (by omega)
  have getNext : (partialRow row below cursor)[cursor + 1]? = some below[cursor + 1] := by
    rw [partialRow_suffix row below cursor (cursor + 1) complete inside.le (by omega)]
    exact List.getElem?_eq_getElem (by omega)
  apply Evaluates.binary (bv := number (max below[cursor] below[cursor + 1]))
    (afterRight := working base (partialRow row below cursor))
    (index_evaluates program _ env "row" rowAddress cursor row _ (.name "j") localRow
      (Evaluates.lookup localCursor) (by intros a b c impossible; cases impossible)
      (prefix_found base _ rowAddress _ getRow) (List.getElem?_eq_getElem inside)) ?_ rfl
  apply Evaluates.builtin (.cons
    (index_evaluates program _ env "dp" base.length cursor _ _ (.name "j") localBelow
      (Evaluates.lookup localCursor) (by intros a b c impossible; cases impossible) (found base _) getCurrent)
    (.cons (index_evaluates program _ env "dp" base.length (cursor + 1) _ _ _ localBelow
      nextIndex (by intros a b c impossible; cases impossible) (found base _) getNext) .nil))
  by_cases larger : below[cursor] < below[cursor + 1]
  · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom, scalarCompare,
      evalCmpOp, HeapCalls.scalarResult, List.mapM, List.mapM.loop, larger,
      max_eq_right (le_of_lt larger)]
  · simp [HeapCalls.primitive, Calls.positionalMaximum, Calls.maximumFrom, scalarCompare,
      evalCmpOp, HeapCalls.scalarResult, List.mapM, List.mapM.loop, larger,
      max_eq_left (le_of_not_gt larger)]

theorem update_result (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (rowAddress cursor : Nat) (row below : List Int)
    (getRow : base[rowAddress]? = some (.list (encode row)))
    (complete : row.length < below.length) (inside : cursor < row.length)
    (localRow : env "row" = some (.ref rowAddress)) (localBelow : env "dp" = some (.ref base.length))
    (localCursor : env "j" = some (number cursor)) :
    BlockContinues program env body (working base (partialRow row below cursor)) env
      (working base (partialRow row below (cursor + 1))) := by
  have inTotals : cursor < (rowTotals row below).length := by
    rw [rowTotals_length_of_complete row below complete]
    exact inside
  have value : (rowTotals row below)[cursor] = row[cursor] + max below[cursor] below[cursor + 1] := by
    have same := rowTotals_getElem? row below cursor
    simpa [List.getElem?_eq_getElem inTotals, List.getElem?_eq_getElem inside,
      List.getElem?_eq_getElem (show cursor < below.length by omega),
      List.getElem?_eq_getElem (show cursor + 1 < below.length by omega)] using same
  apply BlockContinues.setItem (by intros a b c impossible; cases impossible)
    (element_result program base env rowAddress cursor row below getRow complete inside
      localRow localBelow localCursor)
    (.cons (Evaluates.lookup localBelow) (.cons (Evaluates.lookup localCursor) .nil))
  have write := set_item base (partialRow row below cursor) cursor
    (row[cursor] + max below[cursor] below[cursor + 1])
    (by rw [partialRow_length row below cursor complete inside.le]; omega)
  rw [← value, partialRow_step row below cursor complete inside] at write
  simpa only [value] using write

theorem row_loop (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (rowAddress cursor : Nat) (row below : List Int)
    (getRow : base[rowAddress]? = some (.list (encode row)))
    (complete : row.length < below.length) (bounded : cursor ≤ row.length)
    (localRow : env "row" = some (.ref rowAddress)) (localBelow : env "dp" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "j" cursor row.length 1 body []]
      (working base (partialRow row below cursor)) finalEnv
      (working base (partialRow row below row.length)) ∧
      ∀ name, name ≠ "j" → finalEnv name = env name := by
  by_cases active : cursor < row.length
  · let indexed := setLocal env "j" (number cursor)
    have step := update_result program base indexed rowAddress cursor row below getRow complete active
      (by simpa [indexed, setLocal] using localRow) (by simpa [indexed, setLocal] using localBelow) rfl
    obtain ⟨finalEnv, rest, kept⟩ := row_loop program base indexed rowAddress (cursor + 1)
      row below getRow complete (by omega)
      (by simpa [indexed, setLocal] using localRow) (by simpa [indexed, setLocal] using localBelow)
    refine ⟨finalEnv, ?_, ?_⟩
    · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using step.append rest
    · intro name different
      rw [kept name different]
      simp [indexed, setLocal, different]
  · have same : cursor = row.length := by omega
    subst cursor
    exact ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil, by intros; rfl⟩
termination_by row.length - cursor
decreasing_by omega

theorem row_for (program : Module) (base : Mutable.Heap) (env : HeapCalls.Env)
    (rowAddress : Nat) (row below : List Int)
    (getRow : base[rowAddress]? = some (.list (encode row))) (complete : row.length < below.length)
    (localRow : env "row" = some (.ref rowAddress)) (localBelow : env "dp" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env
      [.forIn (.name "j") (.call (.name "range") [.builtinCall .len [.name "row"]]) body []]
      (working base below) finalEnv (working base (partialRow row below row.length)) ∧
      ∀ name, name ≠ "j" → finalEnv name = env name := by
  obtain ⟨finalEnv, loop, kept⟩ := row_loop program base env rowAddress 0 row below getRow complete
    (by omega) localRow localBelow
  refine ⟨finalEnv, ?_, kept⟩
  apply BlockContinues.for_range (values := [number row.length]) (middle := working base below)
    (.cons (Evaluates.builtin (.cons (Evaluates.lookup localRow) .nil)
      (by simp [HeapCalls.primitive, Mutable.length, prefix_found base below rowAddress _ getRow, encode])) .nil) rfl
  simpa only [partialRow_zero] using loop

end ProjectEulerSolutions.TriangleBufferCalls
