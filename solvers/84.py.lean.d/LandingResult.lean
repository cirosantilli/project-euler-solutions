import solvers.«84.py.lean.d».LandingBackThree
import solvers.«84.py.lean.d».LandingWeightPositive

namespace solvers.«84».resolve_landing.weights
open FormalSW.PyAst
set_option Elab.async false

def Post (base : Mutable.Heap) (data : List CardData) (value : Mutable.Value) (after : Mutable.Heap) : Prop :=
  ∃ root entries extra, value = .ref root ∧ after = base ++ extra ∧ base.length ≤ root ∧
    after[root]? = some (.keyDict (encode entries)) ∧ Valid after entries ∧ Distinct entries ∧ Positive entries ∧
    ∀ landing, (lookup entries landing).getD 0 = total data landing

theorem Post.prefix {base extra : Mutable.Heap} {data : List CardData} {value : Mutable.Value} {after : Mutable.Heap}
    (post : Post (base ++ extra) data value after) : Post base data value after := by
  obtain ⟨root, entries, more, returned, same, fresh, found, valid, distinct, positive, counts⟩ := post
  refine ⟨root, entries, extra ++ more, returned, ?_, ?_, found, valid, distinct, positive, counts⟩
  · simpa [List.append_assoc] using same
  · simp only [List.length_append] at fresh
    exact Nat.le_trans (Nat.le_add_right base.length extra.length) fresh

theorem Post.iteration {base : Mutable.Heap} {data : List CardData} {value : Mutable.Value} {after : Mutable.Heap}
    (post : Post base data value after) :
    ∃ root entries, value = .ref root ∧ after[root]? = some (.keyDict (encode entries)) ∧ Valid after entries ∧
      ∀ landing, (entries.map fun entry => if landing = entry.landing then entry.weight else 0).sum = total data landing := by
  obtain ⟨root, entries, _, returned, _, _, found, valid, distinct, _, counts⟩ := post
  refine ⟨root, entries, returned, found, valid, ?_⟩
  intro landing
  rw [← lookup_sum entries landing distinct, counts]

theorem total_support (data : List CardData) (landing : Int × Bool) (pos : 0 < total data landing) :
    ∃ item ∈ data, item.1 = landing := by
  induction data with
  | nil => simp [total] at pos
  | cons item rest ih =>
      by_cases same : landing = item.1
      · exact ⟨item, by simp, same.symm⟩
      · obtain ⟨other, member, key⟩ := ih (by simpa [total, same] using pos)
        exact ⟨other, by simp [member], key⟩

/-- Every stored entry has positive weight, so exact counts exclude extra destinations. -/
theorem Post.bounded {base : Mutable.Heap} {data : List CardData} {value : Mutable.Value} {after : Mutable.Heap}
    (post : Post base data value after) (maximum : Int)
    (bounds : ∀ item ∈ data, 0 ≤ item.1.1 ∧ item.1.1 < maximum) :
    ∃ root entries, value = .ref root ∧ after[root]? = some (.keyDict (encode entries)) ∧ Valid after entries ∧
      ∀ entry ∈ entries, 0 ≤ entry.landing.1 ∧ entry.landing.1 < maximum ∧ 0 < entry.weight := by
  obtain ⟨root, entries, _, returned, _, _, found, valid, distinct, positive, counts⟩ := post
  refine ⟨root, entries, returned, found, valid, ?_⟩
  intro entry member
  have pos := positive entry member
  have count := counts entry.landing
  rw [lookup_at entries entry member distinct] at count
  have present := total_support data entry.landing (by simpa only [Option.getD_some, ← count] using pos)
  obtain ⟨item, allowed, key⟩ := present
  have bound := bounds item allowed
  rw [key] at bound
  exact ⟨bound.1, bound.2, pos⟩

end solvers.«84».resolve_landing.weights
