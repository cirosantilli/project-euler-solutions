import lang.py.FormalSW.HeapMemoCallRules
import lang.py.FormalSW.KeyRelocation

/-! Immutable compiler tuples retain their shared references while exposing fixed contents. -/
namespace ProjectEulerSolutions.TuplePool
open FormalSW.PyAst
set_option Elab.async false

mutual
  def View (heap : Mutable.Heap) : HashKey → Mutable.Value → Prop
    | .scalar constant, value => value = .scalar constant
    | .tuple keys, value => ∃ address items, value = .ref address ∧
        heap[address]? = some (.tuple items) ∧ Views heap keys items
  def Views (heap : Mutable.Heap) : List HashKey → List Mutable.Value → Prop
    | [], [] => True
    | key :: keys, value :: values => View heap key value ∧ Views heap keys values
    | _, _ => False
end

theorem views_scalars (heap : Mutable.Heap) (constants : List Constant) (values : List Mutable.Value) :
    Views heap (constants.map HashKey.scalar) values ↔ values = constants.map Mutable.Value.scalar := by
  induction constants generalizing values with
  | nil => cases values <;> simp [Views]
  | cons constant constants ih => cases values <;> simp [Views, View, ih]

theorem view_scalars (heap : Mutable.Heap) (constants : List Constant) (value : Mutable.Value) :
    View heap (.tuple (constants.map HashKey.scalar)) value ↔
      ∃ address, value = .ref address ∧ heap[address]? = some (.tuple (constants.map Mutable.Value.scalar)) := by
  simp only [View, views_scalars]
  constructor
  · rintro ⟨address, items, rfl, found, rfl⟩
    exact ⟨address, rfl, found⟩
  · rintro ⟨address, rfl, found⟩
    exact ⟨address, _, rfl, found, rfl⟩

mutual
  theorem View.frame (before after : Mutable.Heap)
      (kept : ∀ (address : Nat) (items : List Mutable.Value), before[address]? = some (.tuple items) → after[address]? = some (.tuple items))
      (key : HashKey) (value : Mutable.Value) (view : View before key value) : View after key value := by
    cases key with
    | scalar constant => exact view
    | tuple keys =>
        obtain ⟨address, items, rfl, found, contents⟩ := view
        exact ⟨address, items, rfl, kept address items found, Views.frame before after kept keys items contents⟩
  theorem Views.frame (before after : Mutable.Heap)
      (kept : ∀ (address : Nat) (items : List Mutable.Value), before[address]? = some (.tuple items) → after[address]? = some (.tuple items))
      (keys : List HashKey) (values : List Mutable.Value) (views : Views before keys values) : Views after keys values := by
    cases keys with
    | nil => cases values <;> exact views
    | cons key keys =>
        cases values with
        | nil => exact views
        | cons value values => exact ⟨View.frame before after kept key value views.1, Views.frame before after kept keys values views.2⟩
end

theorem View.append {heap : Mutable.Heap} {key : HashKey} {value : Mutable.Value}
    (view : View heap key value) (extra : Mutable.Heap) : View (heap ++ extra) key value :=
  View.frame heap (heap ++ extra)
    (fun _ _ found => (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found) key value view

mutual
  theorem View.resolve (heap : Mutable.Heap) (key : HashKey) (value : Mutable.Value)
      (view : View heap key value) (fuel : Nat) (enough : key.depth ≤ fuel) :
      Mutable.keyOfFuel heap fuel value = some (.ok key) := by
    cases key with
    | scalar constant => cases view; cases fuel <;> rfl
    | tuple keys =>
        obtain ⟨address, items, rfl, found, contents⟩ := view
        cases fuel with
        | zero => simp [HashKey.depth] at enough
        | succ fuel =>
            have resolved := Views.resolve heap keys items contents fuel (by simpa [HashKey.depth] using enough)
            rw [Mutable.Relocation.keyOfFuel_ref]
            simp [found, resolved, Except.map]
  theorem Views.resolve (heap : Mutable.Heap) (keys : List HashKey) (values : List Mutable.Value)
      (views : Views heap keys values) (fuel : Nat) (enough : HashKey.listDepth keys ≤ fuel) :
      Mutable.Relocation.collectResolved (Mutable.keyOfFuel heap fuel) values = some (.ok keys) := by
    cases keys with
    | nil => cases values with
      | nil => rfl
      | cons value values => exact False.elim views
    | cons key keys =>
        cases values with
        | nil => exact False.elim views
        | cons value values =>
            have head := View.resolve heap key value views.1 fuel (Nat.le_trans (Nat.le_max_left _ _) enough)
            have tail := Views.resolve heap keys values views.2 fuel (Nat.le_trans (Nat.le_max_right _ _) enough)
            simp [Mutable.Relocation.collectResolved, head, tail, Except.map]
end

theorem View.hash {heap : Mutable.Heap} {key : HashKey} {value : Mutable.Value} (view : View heap key value) :
    Mutable.hashKey value heap = some (.ok key, heap) := by
  have resolved := View.resolve heap key value view (max heap.length key.depth) (Nat.le_max_right _ _)
  rw [Mutable.Relocation.keyOfFuel_eq_keyOf heap _ value (Nat.le_max_left _ _)] at resolved
  simp [Mutable.hashKey, resolved]

abbrev Catalog := String → Option (List HashKey)

def spec (catalog : Catalog) (heap : Mutable.Heap) (key : HeapMemoCalls.Key) (value : Mutable.Value) : Prop :=
  ∃ identity keys, catalog identity = some keys ∧ key = HeapMemoCalls.literalKey identity ∧ View heap (.tuple keys) value

theorem preserves_frame (catalog : Catalog) (before after : Mutable.Heap)
    (kept : ∀ (address : Nat) (items : List Mutable.Value), before[address]? = some (.tuple items) → after[address]? = some (.tuple items)) :
    HeapMemoCalls.Cache.Preserves (spec catalog) before after := by
  intro cache valid key value cached
  obtain ⟨identity, keys, known, same, view⟩ := valid key value cached
  exact ⟨identity, keys, known, same, View.frame before after kept _ _ view⟩

theorem preserves_append (catalog : Catalog) (heap extra : Mutable.Heap) :
    HeapMemoCalls.Cache.Preserves (spec catalog) heap (heap ++ extra) :=
  preserves_frame catalog heap (heap ++ extra)
    (fun _ _ found => (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found)

def Post (base : Mutable.Heap) (key : HashKey) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ extra, after = base ++ extra ∧ View after key value

def ArgumentsPost (base : Mutable.Heap) (keys : List HashKey) (values : List Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ extra, after = base ++ extra ∧ Views after keys values

theorem scalar (catalog : Catalog) (module : Module) (env : HeapCalls.Env) (constant : Constant)
    (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.EvaluatesRel module (spec catalog) env (.constant constant) heap (Post heap (.scalar constant)) globals :=
  HeapMemoCalls.EvaluatesRel.constant ⟨[], by simp, rfl⟩

theorem arguments_nil (catalog : Catalog) (module : Module) (env : HeapCalls.Env)
    (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ArgumentsRel module (spec catalog) env [] heap (ArgumentsPost heap []) globals :=
  HeapMemoCalls.ArgumentsRel.nil ⟨[], by simp, trivial⟩

theorem arguments_cons (catalog : Catalog) (module : Module) (env : HeapCalls.Env)
    (expression : Expr) (expressions : List Expr) (key : HashKey) (keys : List HashKey)
    (heap : Mutable.Heap) (globals : Mutable.Globals)
    (head : HeapMemoCalls.EvaluatesRel module (spec catalog) env expression heap (Post heap key) globals)
    (tail : ∀ middle, HeapMemoCalls.ArgumentsRel module (spec catalog) env expressions middle (ArgumentsPost middle keys) globals) :
    HeapMemoCalls.ArgumentsRel module (spec catalog) env (expression :: expressions) heap (ArgumentsPost heap (key :: keys)) globals := by
  apply HeapMemoCalls.ArgumentsRel.cons head
  rintro value middle ⟨extra, rfl, view⟩ operation done anchors stack
  apply (tail (heap ++ extra) operation done anchors stack).consequence
  rintro final ⟨values, after, rfl, more, rfl, views⟩
  exact ⟨values, (heap ++ extra) ++ more, rfl, extra ++ more, by simp [List.append_assoc], view.append more, views⟩

theorem tuple (catalog : Catalog) (module : Module) (env : HeapCalls.Env) (expressions : List Expr)
    (keys : List HashKey) (nonempty : keys ≠ []) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (arguments : HeapMemoCalls.ArgumentsRel module (spec catalog) env expressions heap (ArgumentsPost heap keys) globals) :
    HeapMemoCalls.EvaluatesRel module (spec catalog) env (.tuple expressions) heap (Post heap (.tuple keys)) globals := by
  apply HeapMemoCalls.EvaluatesRel.tuple arguments
  rintro values after ⟨extra, rfl, views⟩
  have populated : values ≠ [] := by
    intro empty
    subst values
    cases keys with
    | nil => exact nonempty rfl
    | cons key keys => exact views
  refine ⟨populated, preserves_append catalog _ _, extra ++ [.tuple values], by simp [List.append_assoc],
    (heap ++ extra).length, values, rfl, by simp, ?_⟩
  exact Views.frame (heap ++ extra) ((heap ++ extra) ++ [.tuple values])
    (fun _ _ found => (List.getElem?_append_left (List.getElem?_eq_some_iff.mp found).1).trans found) keys values views

theorem scalar_arguments (catalog : Catalog) (module : Module) (env : HeapCalls.Env) (constants : List Constant)
    (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ArgumentsRel module (spec catalog) env (constants.map Expr.constant) heap
      (ArgumentsPost heap (constants.map HashKey.scalar)) globals := by
  induction constants generalizing heap with
  | nil => exact arguments_nil catalog module env heap globals
  | cons constant constants ih =>
      exact arguments_cons catalog module env _ _ _ _ heap globals (scalar catalog module env constant heap globals) ih

theorem scalars (catalog : Catalog) (module : Module) (env : HeapCalls.Env) (constants : List Constant)
    (nonempty : constants ≠ []) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.EvaluatesRel module (spec catalog) env (.tuple (constants.map Expr.constant)) heap
      (Post heap (.tuple (constants.map HashKey.scalar))) globals :=
  tuple catalog module env _ _ (by simpa) heap globals (scalar_arguments catalog module env constants heap globals)

/-- A catalog entry constrains both cache hits and first-use tuple allocation. -/
theorem pooled (catalog : Catalog) (identity : String) (keys : List HashKey) (known : catalog identity = some keys)
    (module : Module) (env : HeapCalls.Env) (expressions : List Expr) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (allocate : HeapMemoCalls.EvaluatesRel module (spec catalog) env (.tuple expressions) heap (Post heap (.tuple keys)) globals) :
    HeapMemoCalls.EvaluatesRel module (spec catalog) env (HeapMemoCalls.pooledTuple identity expressions)
      heap (Post heap (.tuple keys)) globals := by
  apply HeapMemoCalls.EvaluatesRel.pooled_tuple
  · apply allocate.consequence
    intro value after post
    exact ⟨post, identity, keys, known, rfl, post.choose_spec.2⟩
  · intro value cached
    obtain ⟨other, otherKeys, found, same, view⟩ := cached
    have equal : identity = other := by
      have names := congrArg MemoCalls.Key.name same
      simpa [HeapMemoCalls.literalKey] using names
    subst other
    have equalKeys : keys = otherKeys := Option.some.inj (known.symm.trans found)
    subst otherKeys
    exact ⟨[], by simp, view⟩

end ProjectEulerSolutions.TuplePool
