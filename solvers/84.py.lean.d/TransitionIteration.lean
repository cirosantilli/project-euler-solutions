import solvers.«84.py.lean.d».TransitionItemsModel
import solvers.«84.py.lean.d».TransitionJail

namespace solvers.«84».transition_row
open FormalSW.PyAst HeapCalls ProjectEulerSolutions NaturalDictionary resolve_landing
set_option Elab.async false

/-- Both actual roll branches add the model contribution and preserve every other object. -/
theorem iteration_result (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (tableRoot root pos move doubles count : Nat) (isDouble : Bool) (entries : Entries)
    (table : Table heap tableRoot) (found : Found heap root entries)
    (getTable : env "landings" = some (.ref tableRoot)) (getRow : env "row" = some (.ref root))
    (getPos : env "pos" = some (number pos)) (getMove : env "move" = some (number move))
    (getDoubles : env "doubles" = some (number doubles)) (getCount : env "count" = some (number count))
    (getDouble : env "is_double" = some (.scalar (.bool isDouble))) :
    ∃ (result : Entries) (finalHeap : Mutable.Heap) (finalEnv : HeapCalls.Env),
      BlockContinues program env code heap finalEnv finalHeap globals ∧ Found finalHeap root result ∧
      items.HeapFrame heap finalHeap root ∧ items.LocalsFrame env finalEnv ∧
      ∀ target, ((lookup result target).getD 0 : Int) =
        ((lookup entries target).getD 0 : Int) + (count : Int) * rowWeight pos doubles move isDouble target := by
  by_cases jail : isDouble = true ∧ doubles = 2
  · obtain ⟨rfl, rfl⟩ := jail
    refine ⟨add entries 30 (256 * count), heap.set root (.dict (encode (add entries 30 (256 * count)))),
      setLocal env "target" (number 30),
      jail_result env heap globals root count entries found getRow getCount getDouble getDoubles,
      found_set found _, ?_, ?_, ?_⟩
    · simpa using items.heap_frame heap [] root (add entries 30 (256 * count))
    · intro name keep
      have different : name ≠ "target" := fun same => keep (by simp [same, items.rebound])
      simp [setLocal, different]
    · intro target
      rw [lookup_add, rowWeight_jail]
      by_cases selected : target = 30 <;> simp [selected, Int.mul_comm]
  · obtain ⟨dictionary, finalHeap, finalEnv, view, ran, finalFound, frame, localFrame⟩ :=
      ordinary_result env heap globals tableRoot root pos move doubles count isDouble entries jail table found
        getTable getRow getPos getMove getDoubles getCount getDouble
    exact ⟨dictionary.entries.foldl (items.applyEntry doubles count isDouble) entries, finalHeap, finalEnv,
      ran, finalFound, frame, localFrame, fun target => ordinary_lookup dictionary heap pos move doubles count target isDouble entries jail view⟩

end solvers.«84».transition_row
