import «.formalsw».solvers.«70.py.ast»
import lang.py.FormalSW.HeapCalls
import ProjectEulerStatements.Lib.TotientLinearSieve

/-! The live lists and alias-preserving writes of the linear totient sieve. -/
namespace solvers.«70».sieve
open FormalSW.PyAst HeapCalls ProjectEulerStatements.TotientLinearSieve
set_option Elab.async false

abbrev program := Solvers.P70Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)

def entries (size : Nat) (value : Nat → Mutable.Value) : List Mutable.Value :=
  (List.range size).map value

/-- Composite marks are integer ones; untouched entries are Boolean false. -/
def markValue (marked : Bool) : Mutable.Value := if marked then number 1 else .scalar (.bool false)

def working (base : Mutable.Heap) (size : Nat) (values : Nat → Nat) (marked : Nat → Bool)
    (cache : List Nat) : Mutable.Heap :=
  base ++ [.list [number 0], .list (entries size (fun n => number (values n))),
    .list [.scalar (.bool false)], .list (entries size (fun n => markValue (marked n))),
    .list (cache.map number)]

abbrev realize (base : Mutable.Heap) (size : Nat) (table : Table) (cache : List Nat) :=
  working base size (fun n => (table n).1) (fun n => (table n).2) cache

theorem entries_lookup (size n : Nat) (value : Nat → Mutable.Value) (within : n < size) :
    (entries size value)[n]? = some (value n) := by simp [entries, within]

theorem entries_update (size n : Nat) (value : Nat → Mutable.Value) (replacement : Mutable.Value)
    (within : n < size) :
    (entries size value).set n replacement =
      entries size (Function.update value n replacement) := by
  apply List.ext_getElem?
  intro index
  by_cases bounded : index < size
  · by_cases same : n = index
    · subst index
      rw [List.getElem?_set_self (by simpa [entries] using within),
        entries_lookup size n _ within]
      simp
    · rw [List.getElem?_set_ne same, entries_lookup size index value bounded,
        entries_lookup size index _ bounded]
      simp [Function.update_of_ne (Ne.symm same)]
  · have outside : size ≤ index := by omega
    simp only [List.getElem?_eq_none (by simpa [entries] using outside :
      ((entries size value).set n replacement).length ≤ index),
      List.getElem?_eq_none (by simpa [entries] using outside :
      (entries size (Function.update value n replacement)).length ≤ index)]

theorem phi_found (base : Mutable.Heap) (size : Nat) (values : Nat → Nat)
    (marked : Nat → Bool) (cache : List Nat) :
    (working base size values marked cache)[base.length + 1]? =
      some (.list (entries size (fun n => number (values n)))) := by simp [working]

theorem marks_found (base : Mutable.Heap) (size : Nat) (values : Nat → Nat)
    (marked : Nat → Bool) (cache : List Nat) :
    (working base size values marked cache)[base.length + 3]? =
      some (.list (entries size (fun n => markValue (marked n)))) := by simp [working]

theorem cache_found (base : Mutable.Heap) (size : Nat) (values : Nat → Nat)
    (marked : Nat → Bool) (cache : List Nat) :
    (working base size values marked cache)[base.length + 4]? =
      some (.list (cache.map number)) := by simp [working]

theorem store_phi (base : Mutable.Heap) (size : Nat) (values replacement : Nat → Nat)
    (marked : Nat → Bool) (cache : List Nat) :
    (working base size values marked cache).set (base.length + 1)
      (.list (entries size (fun n => number (replacement n)))) =
        working base size replacement marked cache := by
  simp [working, List.set_append_right]

theorem store_marks (base : Mutable.Heap) (size : Nat) (values : Nat → Nat)
    (marked replacement : Nat → Bool) (cache : List Nat) :
    (working base size values marked cache).set (base.length + 3)
      (.list (entries size (fun n => markValue (replacement n)))) =
        working base size values replacement cache := by
  simp [working, List.set_append_right]

theorem store_cache (base : Mutable.Heap) (size : Nat) (values : Nat → Nat)
    (marked : Nat → Bool) (cache replacement : List Nat) :
    (working base size values marked cache).set (base.length + 4) (.list (replacement.map number)) =
      working base size values marked replacement := by
  simp [working, List.set_append_right]

private theorem scalar_update {α : Type} (f : α → Mutable.Value) (values : Nat → α)
    (n : Nat) (replacement : α) :
    Function.update (fun k => f (values k)) n (f replacement) =
      fun k => f (Function.update values n replacement k) := by
  funext k
  by_cases same : k = n <;> simp [Function.update_apply, same]

theorem set_list (heap : Mutable.Heap) (address n size : Nat) (value : Nat → Mutable.Value)
    (replacement : Mutable.Value) (within : n < size)
    (found : heap[address]? = some (.list (entries size value))) :
    Mutable.setItem (.ref address) (number n) replacement heap =
      some (.ok (.scalar .none), heap.set address
        (.list (entries size (Function.update value n replacement)))) := by
  have nonnegative : ¬ (n : Int) < 0 := by omega
  have bounded : ¬ size ≤ n := by omega
  rw [← entries_update size n value replacement within]
  simp [Mutable.setItem, Mutable.index, found, entries, number, nonnegative, bounded, Except.map]

theorem read_phi (base : Mutable.Heap) (size n : Nat) (values : Nat → Nat)
    (marked : Nat → Bool) (cache : List Nat) (env : HeapCalls.Env) (indexName : String)
    (within : n < size) (getPhi : env "phi" = some (.ref (base.length + 1)))
    (getIndex : env indexName = some (number n)) :
    Evaluates program env (.subscript (.name "phi") (.name indexName))
      (working base size values marked cache) (number (values n))
      (working base size values marked cache) := by
  apply Evaluates.subscript (by intro a b c impossible; cases impossible)
    (Evaluates.lookup getPhi) (Evaluates.lookup getIndex)
  simpa [entries] using Mutable.getItem_list (working base size values marked cache)
    (base.length + 1) n (entries size (fun k => number (values k)))
    (phi_found base size values marked cache) (by simpa [entries] using within)

/-- The subscription updates the existing totient list, preserving all other objects. -/
theorem set_phi (base : Mutable.Heap) (size n value : Nat) (values : Nat → Nat)
    (marked : Nat → Bool) (cache : List Nat) (env : HeapCalls.Env) (indexName : String)
    (expression : Expr) (within : n < size)
    (getPhi : env "phi" = some (.ref (base.length + 1)))
    (getIndex : env indexName = some (number n))
    (rhs : Evaluates program env expression (working base size values marked cache)
      (number value) (working base size values marked cache)) :
    BlockContinues program env [.assign [.subscript (.name "phi") (.name indexName)] expression]
      (working base size values marked cache) env
      (working base size (Function.update values n value) marked cache) := by
  apply BlockContinues.setItem (by intro a b c impossible; cases impossible) rhs
    (.cons (Evaluates.lookup getPhi) (.cons (Evaluates.lookup getIndex) .nil))
  have written := set_list (working base size values marked cache) (base.length + 1) n size
    (fun k => number (values k)) (number value) within (phi_found base size values marked cache)
  simpa only [scalar_update, store_phi] using written

/-- Marking a composite stores the source integer one without replacing the list. -/
theorem set_mark (base : Mutable.Heap) (size n : Nat) (values : Nat → Nat)
    (marked : Nat → Bool) (cache : List Nat) (env : HeapCalls.Env)
    (within : n < size) (getMarks : env "is_comp" = some (.ref (base.length + 3)))
    (getIndex : env "ip" = some (number n)) :
    BlockContinues program env
      [.assign [.subscript (.name "is_comp") (.name "ip")] (.constant (.int 1))]
      (working base size values marked cache) env
      (working base size values (Function.update marked n true) cache) := by
  apply BlockContinues.setItem (by intro a b c impossible; cases impossible) Evaluates.constant
    (.cons (Evaluates.lookup getMarks) (.cons (Evaluates.lookup getIndex) .nil))
  have written := set_list (working base size values marked cache) (base.length + 3) n size
    (fun k => markValue (marked k)) (markValue true) within
    (marks_found base size values marked cache)
  simpa only [scalar_update, store_marks] using written

/-- The finite heap representation observes only in-range mathematical cells. -/
theorem realize_congr (base : Mutable.Heap) (size : Nat) (left right : Table) (cache : List Nat)
    (same : ∀ n, n < size → left n = right n) :
    realize base size left cache = realize base size right cache := by
  have values : entries size (fun n => number (left n).1) =
      entries size (fun n => number (right n).1) := by
    apply List.map_congr_left
    intro n member
    rw [same n (List.mem_range.mp member)]
  have marks : entries size (fun n => markValue (left n).2) =
      entries size (fun n => markValue (right n).2) := by
    apply List.map_congr_left
    intro n member
    rw [same n (List.mem_range.mp member)]
  simp only [realize, working, values, marks]

end solvers.«70».sieve
