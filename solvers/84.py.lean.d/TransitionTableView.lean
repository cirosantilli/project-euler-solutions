import solvers.«84.py.lean.d».TransitionRowBuild

namespace solvers.«84».transitions
open FormalSW.PyAst ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

def weight (sides : Nat) (state : Nat × Nat) (target : Nat) : Int :=
  ((Monopoly.turnWeights sides state).map fun entry =>
    if target = Monopoly.encode entry.1.1 entry.1.2 then (entry.2 : Int) else 0).sum

structure Row where
  address : Nat
  entries : Entries

def Row.View (row : Row) (sides : Nat) (heap : Mutable.Heap) (state : Nat × Nat) : Prop :=
  Found heap row.address row.entries ∧ ∀ target, ((lookup row.entries target).getD 0 : Int) = weight sides state target

def references (rows : List Row) : List Mutable.Value := rows.map fun row => .ref row.address
def Views (sides : Nat) (heap : Mutable.Heap) (states : List (Nat × Nat)) (rows : List Row) : Prop :=
  List.Forall₂ (fun state row => row.View sides heap state) states rows

theorem Views.frame {sides : Nat} {before after : Mutable.Heap} {states : List (Nat × Nat)} {rows : List Row}
    (views : Views sides before states rows)
    (kept : ∀ (address : Nat) (entries : Entries), Found before address entries → Found after address entries) :
    Views sides after states rows := by
  induction views with
  | nil => exact .nil
  | cons head tail ih => exact .cons ⟨kept _ _ head.1, head.2⟩ ih

theorem Views.append {sides : Nat} {before : Mutable.Heap} {states : List (Nat × Nat)} {rows : List Row}
    (views : Views sides before states rows) (extra : Mutable.Heap) : Views sides (before ++ extra) states rows :=
  views.frame (fun _ _ found => found_append found extra)

theorem Views.snoc {sides : Nat} {heap : Mutable.Heap} {states : List (Nat × Nat)} {rows : List Row}
    (views : Views sides heap states rows) (state : Nat × Nat) (row : Row) (view : row.View sides heap state) :
    Views sides heap (states ++ [state]) (rows ++ [row]) := by
  induction views with
  | nil => exact .cons view .nil
  | cons head tail ih => exact .cons head ih

theorem Views.store {sides : Nat} {heap : Mutable.Heap} {states : List (Nat × Nat)} {rows : List Row}
    (views : Views sides heap states rows) (root : Nat) (old replacement : List Mutable.Value)
    (found : heap[root]? = some (.list old)) : Views sides (heap.set root (.list replacement)) states rows := by
  apply views.frame
  intro address entries present
  have different : root ≠ address := by
    intro same
    rw [same, present] at found
    cases found
  exact (List.getElem?_set_ne different).trans present

abbrev working (base : Mutable.Heap) (rows : List Row) (garbage : Mutable.Heap) : Mutable.Heap :=
  HeapCollection.working base (references rows) garbage

structure Ready (env : HeapCalls.Env) (source tableRoot collector : Nat) : Prop where
  getSource : env "outcomes" = some (.ref source)
  getTable : env "landings" = some (.ref tableRoot)
  getCollector : env "transitions" = some (.ref collector)

def rowBody : List Stmt := transition_row.loopBody (transition_row.loopBody (executableBody outcomes.definition.2) 5) 0

theorem row_source : rowBody = transition_row.initialCode ++
    [.expr (.call (.attribute (.name "transitions") "append") [.name "row"])] := rfl

end solvers.«84».transitions
