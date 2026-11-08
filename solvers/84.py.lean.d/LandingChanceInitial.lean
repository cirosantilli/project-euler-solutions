import solvers.«84.py.lean.d».LandingChanceSetup
import solvers.«84.py.lean.d».LandingWeightUpdates
import solvers.«84.py.lean.d».LandingWeightPositive

namespace solvers.«84».resolve_landing.chance
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

def initialEntries (square : Int) (stay jail : Nat) : List weights.Entry :=
  [⟨(square, false), stay, 96⟩, ⟨(10, true), jail, 16⟩]

def InitialPost (base : Mutable.Heap) (square : Int) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ root stay jail extra, value = .ref root ∧ after = base ++ extra ∧ base.length ≤ root ∧
    after[root]? = some (.keyDict (weights.encode (initialEntries square stay jail))) ∧
    weights.Valid after (initialEntries square stay jail) ∧ weights.Distinct (initialEntries square stay jail) ∧
    weights.Positive (initialEntries square stay jail)

theorem initial_result (square : Int) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (getSquare : env "square" = some (.scalar (.int square))) :
    HeapMemoCalls.EvaluatesRel program pool.spec env display heap (InitialPost heap square) globals := by
  apply (dictionary_result square env heap globals getSquare).consequence
  rintro value after ⟨head, tail, root, extra, correspondence, returned, same, fresh, found, views⟩
  have first := List.forall₂_cons.mp correspondence
  obtain ⟨jail, rest, second, empty, tailSame⟩ := List.forall₂_cons_left_iff.mp first.2
  have restSame := List.forall₂_nil_left_iff.mp empty
  subst rest
  subst tail
  have firstKey := first.1.1
  have firstValue := first.1.2
  have secondKey := second.1
  have secondValue := second.2
  have different : HashKey.equal (.tuple (pool.pair square false)) (.tuple (pool.pair 10 true)) = false := by
    rw [weights.key_equal (square, false) (10, true)]
    simp
  refine ⟨root, head.keyAddress, jail.keyAddress, extra, returned, same, fresh, ?_, ?_, ?_, ?_⟩
  · simpa [List.foldl, TupleDictionary.insert, TupleDictionary.Row.entry, firstKey, firstValue,
      secondKey, secondValue, HashDict.insert, different, weights.encode, initialEntries] using found
  · intro entry member
    simp only [initialEntries, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl
    · simpa only [firstKey] using (views head (by simp)).2
    · simpa only [secondKey] using (views jail (by simp)).2
  · simp [weights.Distinct, initialEntries]
  · simp [weights.Positive, initialEntries]

end solvers.«84».resolve_landing.chance
