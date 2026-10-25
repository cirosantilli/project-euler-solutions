import solvers.«67.inputs.source»
import ProjectEulerSolutions.Lib.TriangleBufferCalls

namespace solvers.«67»
open FormalSW.PyAst HeapCalls ProjectEulerStatements.P18
open ProjectEulerSolutions.TriangleBufferCalls (working number)
set_option Elab.async false

abbrev triangleBody : List Stmt :=
  [.assign [.name "row"] (.subscript (.name "triangle") (.name "i")),
   .forIn (.name "j") (.call (.name "range") [.builtinCall .len [.name "row"]])
     ProjectEulerSolutions.TriangleBufferCalls.body []]
abbrev finalReturn : Stmt := .return (.subscript (.name "dp") (.constant (.int 0)))

theorem triangle_loop (remaining : Nat) (base : Mutable.Heap) (env : HeapCalls.Env)
    (address : Nat) (rows : TriangleRows) (buffer : List Int)
    (getTriangle : base[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, base[pair.1]? = some (.list (encode pair.2)))
    (shape : RowShape rows) (bounded : remaining ≤ rows.length) (room : remaining < buffer.length)
    (localTriangle : env "triangle" = some (.ref address))
    (localBuffer : env "dp" = some (.ref base.length)) :
    ∃ finalBuffer, BlockReturns program env
      [.rangeStepLoop "i" ((remaining : Int) - 1) (-1) (-1) triangleBody [], finalReturn]
      (working base buffer)
      (number ((reducePrefix (rows.map Prod.snd) remaining (buffer.take (remaining + 1))).headD 0))
      (working base finalBuffer) := by
  induction remaining generalizing env buffer with
  | zero =>
      refine ⟨buffer, ?_⟩
      apply BlockReturns.range_done rfl
      apply BlockReturns.return_value
      have same : ((buffer.take 1).headD 0) = buffer[0]'room := by
        cases buffer with
        | nil => simp at room
        | cons _ _ => rfl
      change Evaluates program env (.subscript (.name "dp") (.constant (.int 0)))
        (working base buffer) (number ((buffer.take 1).headD 0)) (working base buffer)
      rw [same]
      exact ProjectEulerSolutions.TriangleBufferCalls.index_evaluates program (working base buffer) env
          "dp" base.length 0 buffer _ (.constant (.int 0)) localBuffer Evaluates.constant
          (by intros a b c impossible; cases impossible)
          (ProjectEulerSolutions.TriangleBufferCalls.found base buffer) (List.getElem?_eq_getElem room)
  | succ remaining ih =>
      have valid : remaining < rows.length := by omega
      let pair := rows[remaining]
      have rowSize : pair.2.length = remaining + 1 := shape remaining valid
      let indexed := setLocal env "i" (number remaining)
      let rowEnv := setLocal indexed "row" (.ref pair.1)
      have readRow : Evaluates program indexed (.subscript (.name "triangle") (.name "i"))
          (working base buffer) (.ref pair.1) (working base buffer) := by
        apply Evaluates.subscript (by intros a b c impossible; cases impossible)
          (Evaluates.lookup (by simpa [indexed, setLocal] using localTriangle)) (Evaluates.lookup rfl)
        have read := Mutable.getItem_list (working base buffer) address remaining
          (rows.map (fun pair => .ref pair.1))
          (ProjectEulerSolutions.TriangleBufferCalls.prefix_found base buffer address _ getTriangle)
          (by simpa using valid)
        rw [List.getElem_map] at read
        exact read
      obtain ⟨finalEnv, calculated, kept⟩ := ProjectEulerSolutions.TriangleBufferCalls.row_for
        program base rowEnv pair.1 pair.2 buffer (getRows pair (List.getElem_mem valid)) (by omega)
        rfl (by simpa [rowEnv, indexed, setLocal] using localBuffer)
      let updated := partialRow pair.2 buffer pair.2.length
      have updatedSize : updated.length = buffer.length :=
        partialRow_length pair.2 buffer pair.2.length (by omega) (by omega)
      obtain ⟨finalBuffer, continued⟩ := ih finalEnv updated (by omega) (by omega)
        (by rw [kept "triangle" (by decide)]; simpa [rowEnv, indexed, setLocal] using localTriangle)
        (by rw [kept "dp" (by decide)]; simpa [rowEnv, indexed, setLocal] using localBuffer)
      have same : reducePrefix (rows.map Prod.snd) (remaining + 1) (buffer.take (remaining + 2)) =
          reducePrefix (rows.map Prod.snd) remaining (updated.take (remaining + 1)) := by
        have step := reducePrefix_buffer_step (rows.map Prod.snd) remaining buffer
          (by simpa using valid) (by simpa only [List.getElem_map] using rowSize) (by omega)
        simpa only [List.getElem_map, ← rowSize] using step
      refine ⟨finalBuffer, ?_⟩
      rw [show ((remaining + 1 : Nat) : Int) - 1 = remaining by omega, same]
      apply BlockReturns.range_step (by simp [Calls.rangeContinues]; omega)
      apply BlockReturns.assign readRow
      simpa [triangleBody, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        List.append_assoc, sub_eq_add_neg] using calculated.returns continued

abbrev lastExpression : Expr :=
  .subscript (.name "triangle") (.unaryOp .uSub (.constant (.int 1)))

theorem solve.result (heap : Mutable.Heap) (address : Nat) (rows : TriangleRows)
    (getTriangle : heap[address]? = some (.list (rows.map (fun pair => .ref pair.1))))
    (getRows : ∀ pair ∈ rows, heap[pair.1]? = some (.list (encode pair.2))) (shape : RowShape rows) :
    ∃ extra, ReturnsValue program "solve" [.ref address] heap
      (number ((reduceRowValues (rows.map Prod.snd)).headD 0)) (heap ++ extra) := by
  let args : List Arg := [⟨"triangle", some (.subscript (.name "list")
    (.subscript (.name "list") (.name "int"))), none⟩]
  let env := HeapCalls.argumentEnv args [.ref address]
  have tested : Evaluates program env (.unaryOp .not (.name "triangle")) heap
      (.scalar (.bool (decide (rows = [])))) heap :=
    Evaluates.unary (Evaluates.lookup rfl)
      (by simp [HeapCalls.unary, Mutable.truth, getTriangle]; cases rows <;> rfl)
  by_cases empty : rows = []
  · refine ⟨[], ?_⟩
    apply BlockReturns.function (args := args) rfl rfl
    apply BlockReturns.branch (flag := true) (by simpa only [empty, decide_true] using tested) rfl
    simpa [empty, reduceRowValues, number] using
      (BlockReturns.return_value (module := program) (heap := heap) (env := env) Evaluates.constant
        (value := .scalar (.int 0)))
  · have positive : 0 < rows.length := List.length_pos_iff.mpr empty
    have valid : rows.length - 1 < rows.length := by omega
    let bottom := rows[rows.length - 1]
    let dpEnv := setLocal env "dp" (.ref heap.length)
    have shifted : (-1 + (rows.length : Int)).toNat = rows.length - 1 := by omega
    have indexed : Mutable.index (rows.map (fun pair => Mutable.Value.ref pair.1))
        (.scalar (.int (-1))) heap = some (.ok (rows.length - 1), heap) := by
      simp [Mutable.index, empty, shifted]
    have last : Evaluates program env lastExpression heap (.ref bottom.1) heap := by
      apply Evaluates.subscript (by intros a b c impossible; cases impossible) (Evaluates.lookup rfl)
        (Evaluates.unary Evaluates.constant rfl)
      simp [Mutable.getItem, getTriangle, indexed, Except.map, List.getElem?_eq_getElem valid, bottom]
    have copied : Evaluates program env (.subscript lastExpression (.slice none none none)) heap
        (.ref heap.length) (working heap bottom.2) := by
      exact Evaluates.slice (.cons last (.cons .constant (.cons .constant (.cons .constant .nil))))
        (Mutable.getSlice_list_copy heap bottom.1 (encode bottom.2) (getRows bottom (List.getElem_mem valid)))
    have bottomSize : bottom.2.length = rows.length := by
      have size := shape (rows.length - 1) valid
      dsimp [bottom]
      omega
    obtain ⟨finalBuffer, loop⟩ := triangle_loop (rows.length - 1) heap dpEnv address rows bottom.2
      getTriangle getRows shape (by omega) (by omega) rfl rfl
    have same : reduceRowValues (rows.map Prod.snd) =
        reducePrefix (rows.map Prod.snd) (rows.length - 1)
          (bottom.2.take (rows.length - 1 + 1)) := by
      rw [List.take_of_length_le (by omega : bottom.2.length ≤ rows.length - 1 + 1)]
      simpa only [List.length_map, List.getElem_map] using
        reduceRowValues_from_last (rows.map Prod.snd) (by simpa using empty)
    refine ⟨[.list (ProjectEulerSolutions.TriangleBufferCalls.encode finalBuffer)], ?_⟩
    apply BlockReturns.function (args := args) rfl rfl
    apply BlockReturns.branch (flag := false) (by simpa only [empty, decide_false] using tested) rfl
    apply BlockReturns.assign copied
    apply BlockReturns.for_range (start := ((rows.length - 1 : Nat) : Int) - 1) (stop := -1) (stride := -1)
      (middle := working heap bottom.2)
      (values := [number ((rows.length : Int) - 2), number (-1), number (-1)])
    · apply Evaluates.Arguments.cons
      · exact Evaluates.binary (av := number rows.length) (middle := working heap bottom.2)
          (Evaluates.builtin (.cons (Evaluates.lookup rfl) .nil)
            (by simp [HeapCalls.primitive, Mutable.length,
              ProjectEulerSolutions.TriangleBufferCalls.prefix_found heap bottom.2 address _ getTriangle]))
          Evaluates.constant rfl
      · exact .cons (Evaluates.unary .constant rfl) (.cons (Evaluates.unary .constant rfl) .nil)
    · have start : (rows.length : Int) - 2 = ((rows.length - 1 : Nat) : Int) - 1 := by omega
      simpa only [start] using (show HeapCalls.rangeBounds
        [number (((rows.length - 1 : Nat) : Int) - 1), number (-1), number (-1)] =
        .ok (((rows.length - 1 : Nat) : Int) - 1, -1, -1) from rfl)
    · simpa only [same] using loop

end solvers.«67»
