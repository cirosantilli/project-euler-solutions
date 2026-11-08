import ProjectEulerSolutions.Lib.TupleDictionary
import Mathlib.Data.List.Forall2

namespace ProjectEulerSolutions.TupleDictionary
open FormalSW.PyAst
set_option Elab.async false

abbrev Description := List HashKey × Constant
def Matches (description : Description) (row : Row) : Prop :=
  row.key = .tuple description.1 ∧ row.value = .scalar description.2

theorem row_of_view (heap : Mutable.Heap) (keys : List HashKey) (constant : Constant) (value : Mutable.Value)
    (view : TuplePool.View heap (.tuple [.tuple keys, .scalar constant]) value) :
    ∃ row, value = .ref row.address ∧ Matches (keys, constant) row ∧ row.View heap := by
  obtain ⟨address, items, rfl, found, contents⟩ := view
  cases items with
  | nil => exact False.elim contents
  | cons keyValue items =>
      cases items with
      | nil => exact False.elim contents.2
      | cons stored items =>
          cases items with
          | cons extra items => exact False.elim contents.2.2
          | nil =>
              have storedEq : stored = .scalar constant := contents.2.1
              subst stored
              obtain ⟨keyAddress, values, rfl, keyFound, keyContents⟩ := contents.1
              exact ⟨⟨address, keyAddress, .tuple keys, .scalar constant⟩, rfl, ⟨rfl, rfl⟩,
                found, keyAddress, values, rfl, keyFound, keyContents⟩

def RowPost (base : Mutable.Heap) (description : Description) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ row extra, value = .ref row.address ∧ after = base ++ extra ∧ Matches description row ∧ row.View after

theorem row_result (catalog : TuplePool.Catalog) (module : Module) (env : HeapCalls.Env) (expression : Expr)
    (keys : List HashKey) (constant : Constant) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (key : HeapMemoCalls.EvaluatesRel module (TuplePool.spec catalog) env expression heap (TuplePool.Post heap (.tuple keys)) globals) :
    HeapMemoCalls.EvaluatesRel module (TuplePool.spec catalog) env (.tuple [expression, .constant constant]) heap
      (RowPost heap (keys, constant)) globals := by
  have allocated : HeapMemoCalls.EvaluatesRel module (TuplePool.spec catalog) env (.tuple [expression, .constant constant]) heap
      (TuplePool.Post heap (.tuple [.tuple keys, .scalar constant])) globals := by
    apply TuplePool.tuple catalog module env _ _ (by simp)
    exact TuplePool.arguments_cons catalog module env _ _ _ _ heap globals key
      (fun middle => TuplePool.scalar_arguments catalog module env [constant] middle globals)
  apply allocated.consequence
  rintro value after ⟨extra, same, view⟩
  obtain ⟨row, returned, correspondence, found⟩ := row_of_view after keys constant value view
  exact ⟨row, extra, returned, same, correspondence, found⟩

def ArgumentsPost (base : Mutable.Heap) (descriptions : List Description)
    (values : List Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ rows extra, values = rows.map (fun row => .ref row.address) ∧ after = base ++ extra ∧
    List.Forall₂ Matches descriptions rows ∧ ∀ row ∈ rows, row.View after

theorem arguments_nil (catalog : TuplePool.Catalog) (module : Module) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ArgumentsRel module (TuplePool.spec catalog) env [] heap (ArgumentsPost heap []) globals :=
  HeapMemoCalls.ArgumentsRel.nil ⟨[], [], rfl, by simp, .nil, by simp⟩

theorem arguments_cons (catalog : TuplePool.Catalog) (module : Module) (env : HeapCalls.Env)
    (expression : Expr) (expressions : List Expr) (description : Description) (descriptions : List Description)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (head : HeapMemoCalls.EvaluatesRel module (TuplePool.spec catalog) env expression heap (RowPost heap description) globals)
    (tail : ∀ middle, HeapMemoCalls.ArgumentsRel module (TuplePool.spec catalog) env expressions middle (ArgumentsPost middle descriptions) globals) :
    HeapMemoCalls.ArgumentsRel module (TuplePool.spec catalog) env (expression :: expressions) heap
      (ArgumentsPost heap (description :: descriptions)) globals := by
  apply HeapMemoCalls.ArgumentsRel.cons head
  rintro value middle ⟨row, extra, rfl, rfl, matched, view⟩ operation done anchors stack
  apply (tail (heap ++ extra) operation done anchors stack).consequence
  rintro final ⟨values, after, rfl, rows, more, rfl, rfl, correspondence, views⟩
  refine ⟨_, _, rfl, row :: rows, extra ++ more, rfl, by simp [List.append_assoc], .cons matched correspondence, ?_⟩
  intro current member
  rcases List.mem_cons.mp member with rfl | member
  · exact view.append more
  · exact views current member

def Post (base : Mutable.Heap) (descriptions : List Description) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ head tail root extra, List.Forall₂ Matches descriptions (head :: tail) ∧
    value = .ref root ∧ after = base ++ extra ∧ base.length ≤ root ∧
    after[root]? = some (.keyDict (tail.foldl insert [head.entry])) ∧
    ∀ row ∈ head :: tail, row.View after

theorem Post.prefix {base extra : Mutable.Heap} {descriptions : List Description}
    {value : Mutable.Value} {after : Mutable.Heap} (post : Post (base ++ extra) descriptions value after) :
    Post base descriptions value after := by
  obtain ⟨head, tail, root, more, correspondence, returned, same, fresh, found, views⟩ := post
  refine ⟨head, tail, root, extra ++ more, correspondence, returned, ?_, ?_, found, views⟩
  · simpa [List.append_assoc] using same
  · simp only [List.length_append] at fresh
    exact Nat.le_trans (Nat.le_add_right base.length extra.length) fresh

end ProjectEulerSolutions.TupleDictionary
