import solvers.«84.py.lean.d».OutcomeView

namespace solvers.«84».transition_row.rolls
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary
set_option Elab.async false

def names : List String := ["move", "is_double", "count"]
def locals (env : HeapCalls.Env) (outcome : outcomes.Outcome) : HeapCalls.Env := assignNames env names outcome.items
def loop : Stmt := .forIn (.tuple [.name "move", .name "is_double", .name "count"]) (.name "outcomes") code []
def cursor (source position : Nat) : Stmt := .unpackStepLoop names (.ref source) position code []

theorem source : (loopBody (loopBody (executableBody outcomes.definition.2) 5) 0)[(1 : Nat)]? = some loop := rfl

def total (pos doubles : Nat) (values : List outcomes.Outcome) (target : Nat) : Int :=
  (values.map fun outcome => outcome.count * rowWeight pos doubles outcome.move.toNat outcome.isDouble target).sum

def rebound : List String := names ++ items.rebound
def LocalsFrame (before after : HeapCalls.Env) : Prop := ∀ name, name ∉ rebound → after name = before name

theorem locals_frame (env after : HeapCalls.Env) (outcome : outcomes.Outcome)
    (frame : items.LocalsFrame (locals env outcome) after) : LocalsFrame env after := by
  intro name keep
  have separate : name ∉ names ∧ name ∉ items.rebound := by simpa [rebound] using keep
  rw [frame name separate.2]
  simp only [names, List.mem_cons, not_or] at separate
  simp_all [locals, names, assignNames, setLocal, outcomes.Outcome.items]

theorem step_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tableRoot root pos doubles : Nat) (outcome : outcomes.Outcome) (entries : Entries)
    (table : Table heap tableRoot) (found : Found heap root entries)
    (getTable : env "landings" = some (.ref tableRoot)) (getRow : env "row" = some (.ref root))
    (getPos : env "pos" = some (number pos)) (getDoubles : env "doubles" = some (number doubles))
    (moveNonneg : 0 ≤ outcome.move) (countNonneg : 0 ≤ outcome.count) :
    ∃ (result : Entries) (finalHeap : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program (locals env outcome) code heap finalEnv finalHeap globals ∧ Found finalHeap root result ∧
      items.HeapFrame heap finalHeap root ∧ LocalsFrame env finalEnv ∧
      ∀ target, ((lookup result target).getD 0 : Int) = ((lookup entries target).getD 0 : Int) +
        outcome.count * rowWeight pos doubles outcome.move.toNat outcome.isDouble target := by
  obtain ⟨result, finalHeap, finalEnv, ran, finalFound, frame, localFrame, counts⟩ :=
    iteration_result (locals env outcome) heap globals tableRoot root pos outcome.move.toNat doubles outcome.count.toNat outcome.isDouble entries
      table found
      (by simpa [locals, names, assignNames, setLocal, outcomes.Outcome.items] using getTable)
      (by simpa [locals, names, assignNames, setLocal, outcomes.Outcome.items] using getRow)
      (by simpa [locals, names, assignNames, setLocal, outcomes.Outcome.items] using getPos)
      (by simp [locals, names, assignNames, setLocal, outcomes.Outcome.items, Int.toNat_of_nonneg moveNonneg])
      (by simpa [locals, names, assignNames, setLocal, outcomes.Outcome.items] using getDoubles)
      (by simp [locals, names, assignNames, setLocal, outcomes.Outcome.items, Int.toNat_of_nonneg countNonneg])
      (by simp [locals, names, assignNames, setLocal, outcomes.Outcome.items])
  refine ⟨result, finalHeap, finalEnv, ran, finalFound, frame, locals_frame env finalEnv outcome localFrame, ?_⟩
  simpa only [Int.toNat_of_nonneg countNonneg] using counts

end solvers.«84».transition_row.rolls
