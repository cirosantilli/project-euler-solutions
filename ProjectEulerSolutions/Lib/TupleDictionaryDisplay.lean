import ProjectEulerSolutions.Lib.TupleDictionaryRows

namespace ProjectEulerSolutions.TupleDictionary
open FormalSW.PyAst
set_option Elab.async false

/-- Small source displays evaluate their keys once in source order, including cache hits. -/
theorem display (catalog : TuplePool.Catalog) (module : Module) (env : HeapCalls.Env)
    (expression : Expr) (keys : List HashKey) (constant : Constant)
    (first : Expr × Expr) (rest : List (Expr × Expr)) (small : (first :: rest).length < 15)
    (descriptions : List Description) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (key : HeapMemoCalls.EvaluatesRel module (TuplePool.spec catalog) env expression heap (TuplePool.Post heap (.tuple keys)) globals)
    (tail : ∀ middle, HeapMemoCalls.ArgumentsRel module (TuplePool.spec catalog) env
      ((first :: rest).map fun pair => .tuple [pair.1, pair.2]) middle (ArgumentsPost middle descriptions) globals) :
    HeapMemoCalls.EvaluatesRel module (TuplePool.spec catalog) env
      (.dict ((expression, .constant constant) :: first :: rest)) heap (Post heap ((keys, constant) :: descriptions)) globals := by
  intro anchors stack
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.dictionary ((expression, .constant constant) :: first :: rest) [] env, stack, heap⟩) (by rfl) rfl
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.expr expression env, .dictionaryKey (.constant constant) (first :: rest) [] env :: stack, heap⟩) (by trivial) rfl
  apply (key anchors _).bind
  rintro state ⟨value, middle, rfl, extra, heapSame, view⟩
  have keyView := view
  obtain ⟨keyAddress, keyItems, rfl, keyFound, keyContents⟩ := view
  let headRow : Row := ⟨middle.length, keyAddress, .tuple keys, .scalar constant⟩
  let firstHeap := middle ++ [Mutable.Object.tuple [.ref keyAddress, .scalar constant]]
  let tailCode := (first :: rest).map fun pair => Expr.tuple [pair.1, pair.2]
  let continuation := HeapCalls.Frame.arguments .list [] tailCode env ::
    HeapCalls.Frame.arguments (.builtin .dict) [] [] env :: stack
  have headView : headRow.View firstHeap := ⟨by simp [headRow, firstHeap], keyView.append _⟩
  apply HeapMemoCalls.RunsRel.next
    (following := HeapCalls.dictionaryDisplayContinue (.ref keyAddress) (.constant constant) (first :: rest) env stack middle)
    (by trivial) (by simp only [HeapCalls.step, if_pos small, HeapCalls.protectGenerator_next])
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.value (.scalar constant), .arguments .tuple [.ref keyAddress] [] env :: continuation, middle⟩) (by rfl) rfl
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.operation .tuple [.ref keyAddress, .scalar constant], continuation, middle⟩) (by trivial) rfl
  apply HeapMemoCalls.RunsRel.next
    (following := ⟨.value (.ref middle.length), continuation, firstHeap⟩)
    (preserved := TuplePool.preserves_append catalog middle _) (by trivial)
  · simp [HeapCalls.step, HeapCalls.primitive, HeapCalls.result, HeapCalls.run, Mutable.allocateTuple, firstHeap, Except.map]
  · apply HeapMemoCalls.RunsRel.next
      (following := HeapCalls.argumentsState .list env [.ref middle.length] tailCode
        (.arguments (.builtin .dict) [] [] env :: stack) firstHeap) (by trivial) rfl
    apply (tail firstHeap .list [.ref middle.length] anchors _).bind
    rintro final ⟨values, after, rfl, rows, more, rfl, afterSame, correspondence, views⟩
    let allRows := headRow :: rows
    let rowValues := allRows.map fun row => Mutable.Value.ref row.address
    let withRows := after ++ [Mutable.Object.list rowValues]
    let dictionary := Mutable.Object.keyDict (rows.foldl insert [headRow.entry])
    have firstAfter : headRow.View after := by rw [afterSame]; exact headView.append more
    have allViews : ∀ row ∈ allRows, row.View after := by
      intro row member
      rcases List.mem_cons.mp member with rfl | member
      · exact firstAfter
      · exact views row member
    have built := construct withRows after.length headRow rows (by simp [withRows, rowValues, allRows])
      (fun row member => (allViews row member).append _)
    apply HeapMemoCalls.RunsRel.next
      (following := ⟨.value (.ref after.length), .arguments (.builtin .dict) [] [] env :: stack, withRows⟩)
      (preserved := TuplePool.preserves_append catalog after _) (by trivial)
    · simp [HeapCalls.step, HeapCalls.primitive, HeapCalls.result, HeapCalls.run, Mutable.allocate,
        withRows, rowValues, allRows, headRow, Except.map]
    · apply HeapMemoCalls.RunsRel.next
        (following := ⟨.operation (.builtin .dict) [.ref after.length], stack, withRows⟩) (by trivial) rfl
      apply HeapMemoCalls.RunsRel.next
        (following := ⟨.value (.ref withRows.length), stack, withRows ++ [dictionary]⟩)
        (preserved := TuplePool.preserves_append catalog withRows _) (by trivial)
      · simp [HeapCalls.step, HeapCalls.primitive, HeapCalls.result, HeapCalls.run, built, dictionary]
      · apply HeapMemoCalls.RunsRel.refl
        refine ⟨.ref withRows.length, withRows ++ [dictionary], rfl, headRow, rows, withRows.length,
          extra ++ [.tuple [.ref keyAddress, .scalar constant]] ++ more ++ [.list rowValues, dictionary],
          .cons ⟨rfl, rfl⟩ correspondence, rfl, ?_, ?_, by simp [dictionary], ?_⟩
        · simp [withRows, afterSame, firstHeap, heapSame, List.append_assoc]
        · simp [withRows, afterSame, firstHeap, heapSame]
        · intro row member
          exact ((allViews row member).append [.list rowValues]).append [dictionary]

end ProjectEulerSolutions.TupleDictionary
