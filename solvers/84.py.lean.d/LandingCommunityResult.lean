import solvers.«84.py.lean.d».LandingCommunity
import solvers.«84.py.lean.d».LandingResult

namespace solvers.«84».resolve_landing.community
open FormalSW.PyAst ProjectEulerSolutions
set_option Elab.async false

def data (square : Int) : List weights.CardData := [((square, false), 224), ((0, false), 16), ((10, true), 16)]
def entries (square : Int) (stay go jail : Nat) : List weights.Entry :=
  [⟨(square, false), stay, 224⟩, ⟨(0, false), go, 16⟩, ⟨(10, true), jail, 16⟩]

theorem post_weights (square : Int) (notZero : square ≠ 0) (heap : Mutable.Heap)
    (value : Mutable.Value) (after : Mutable.Heap) (post : TupleDictionary.Post heap (descriptions square) value after) :
    weights.Post heap (data square) value after := by
  obtain ⟨head, tail, root, extra, correspondence, returned, same, fresh, found, views⟩ := post
  have first := List.forall₂_cons.mp correspondence
  obtain ⟨go, rest, second, following, tailSame⟩ := List.forall₂_cons_left_iff.mp first.2
  obtain ⟨jail, last, third, empty, restSame⟩ := List.forall₂_cons_left_iff.mp following
  have lastSame := List.forall₂_nil_left_iff.mp empty
  subst last
  subst rest
  subst tail
  have firstKey := first.1.1
  have firstValue := first.1.2
  have secondKey := second.1
  have secondValue := second.2
  have thirdKey := third.1
  have thirdValue := third.2
  have stayGo : HashKey.equal (.tuple (pool.pair square false)) (.tuple (pool.pair 0 false)) = false := by
    rw [weights.key_equal (square, false) (0, false)]
    simp [notZero]
  have stayJail : HashKey.equal (.tuple (pool.pair square false)) (.tuple (pool.pair 10 true)) = false := by
    rw [weights.key_equal (square, false) (10, true)]
    simp
  have distinct : weights.Distinct (entries square head.keyAddress go.keyAddress jail.keyAddress) := by
    simp [weights.Distinct, entries, notZero]
  refine ⟨root, entries square head.keyAddress go.keyAddress jail.keyAddress, extra, returned, same, fresh, ?_, ?_, distinct, ?_, ?_⟩
  · simpa [List.foldl, TupleDictionary.insert, TupleDictionary.Row.entry, firstKey, firstValue,
      secondKey, secondValue, thirdKey, thirdValue, HashDict.insert, stayGo, stayJail,
      weights.key_equal (0, false) (10, true), weights.encode, entries] using found
  · intro entry member
    simp only [entries, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl
    · simpa only [firstKey] using (views head (by simp)).2
    · simpa only [secondKey] using (views go (by simp)).2
    · simpa only [thirdKey] using (views jail (by simp)).2
  · simp [weights.Positive, entries]
  · intro landing
    rw [weights.lookup_sum _ _ distinct]
    simp [entries, weights.total, data]

theorem weights_result (square : Int) (member : square ∈ [2, 17, 33]) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    HeapMemoCalls.ReturnsRel program pool.spec "resolve_landing" [.scalar (.int square)] heap
      (weights.Post heap (data square)) globals := by
  apply (result square member heap globals).consequence
  apply post_weights square
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  omega

end solvers.«84».resolve_landing.community
