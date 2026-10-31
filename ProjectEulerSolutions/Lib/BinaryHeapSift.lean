import ProjectEulerSolutions.Lib.BinaryHeapOrder

/-! Order invariants for the downward-then-upward repair used by array heaps.
The hole contains the displaced entry. Its parent must already precede its
children, even when neither edge through the hole is currently ordered. -/
namespace ProjectEulerSolutions.BinaryHeapOrder

def parent (i : Nat) : Nat := (i - 1) / 2

theorem parent_lt {i : Nat} (positive : 0 < i) : parent i < i := by
  unfold parent
  have := Nat.div_le_self (i - 1) 2
  omega

def Edges (size : Nat) (weight : Nat → Int) : Prop :=
  ∀ i, i < size → 0 < i → weight (parent i) ≤ weight i

structure Hole (size hole : Nat) (weight : Nat → Int) : Prop where
  edges : ∀ i, i < size → 0 < i → i ≠ hole → parent i ≠ hole →
    weight (parent i) ≤ weight i
  bridge : ∀ i, i < size → 0 < i → parent i = hole → 0 < hole →
    weight (parent hole) ≤ weight i

structure UpHole (size hole : Nat) (weight : Nat → Int) : Prop extends Hole size hole weight where
  children : ∀ i, i < size → 0 < i → parent i = hole → weight hole ≤ weight i

def exchangeWeight (weight : Nat → Int) (a b i : Nat) : Int :=
  if i = b then weight a else if i = a then weight b else weight i

theorem UpHole.edge {size hole : Nat} {weight : Nat → Int}
    (ordered : UpHole size hole weight) (i : Nat) (inside : i < size)
    (positive : 0 < i) (other : i ≠ hole) : weight (parent i) ≤ weight i := by
  by_cases child : parent i = hole
  · simpa only [child] using ordered.children i inside positive child
  · exact ordered.edges i inside positive other child

theorem UpHole.close {size hole : Nat} {weight : Nat → Int}
    (ordered : UpHole size hole weight)
    (stopped : 0 < hole → weight (parent hole) ≤ weight hole) : Edges size weight := by
  intro i inside positive
  by_cases same : i = hole
  · subst i
    exact stopped positive
  · exact ordered.edge i inside positive same

theorem Hole.leaf {size hole : Nat} {weight : Nat → Int}
    (ordered : Hole size hole weight)
    (leaf : ∀ i, i < size → 0 < i → parent i ≠ hole) : UpHole size hole weight := by
  refine ⟨ordered, ?_⟩
  intro i inside positive child
  exact False.elim (leaf i inside positive child)

/-- Moving the smaller child into a hole preserves all edges away from the new hole. -/
theorem Hole.descend {size hole child : Nat} {weight : Nat → Int}
    (ordered : Hole size hole weight) (inside : child < size) (positive : 0 < child)
    (childOf : parent child = hole)
    (least : ∀ i, i < size → 0 < i → parent i = hole → weight child ≤ weight i) :
    Hole size child (exchangeWeight weight hole child) := by
  have smaller : hole < child := childOf ▸ parent_lt positive
  have holeInside : hole < size := by omega
  constructor
  · intro i valid pos notChild parentNotChild
    have pi := parent_lt pos
    by_cases same : i = hole
    · subst i
      have pp := parent_lt pos
      have old := ordered.bridge child inside positive childOf pos
      simpa [exchangeWeight, childOf, (show hole ≠ child by omega), (show parent hole ≠ hole by omega),
        show parent hole ≠ child by omega] using old
    · by_cases parentHole : parent i = hole
      · have old := least i valid pos parentHole
        simpa [exchangeWeight, notChild, same, parentHole, (show hole ≠ child by omega)] using old
      · have old := ordered.edges i valid pos same parentHole
        simpa [exchangeWeight, notChild, same, parentNotChild, parentHole] using old
  · intro i valid pos parentChild _
    have pi := parent_lt pos
    have old := ordered.edges i valid pos (by omega) (by omega)
    simpa [exchangeWeight, childOf, parentChild, (show hole ≠ child by omega),
      show i ≠ child by omega, show i ≠ hole by omega] using old

/-- Upward repair needs the bridge condition: the old parent will replace the hole. -/
theorem UpHole.ascend {size hole : Nat} {weight : Nat → Int}
    (ordered : UpHole size hole weight) (inside : hole < size) (positive : 0 < hole)
    (moving : weight hole ≤ weight (parent hole)) :
    UpHole size (parent hole) (exchangeWeight weight hole (parent hole)) := by
  have smaller := parent_lt positive
  have parentInside : parent hole < size := by omega
  have newEdge (i : Nat) (valid : i < size) (pos : 0 < i) (other : i ≠ parent hole) :
      exchangeWeight weight hole (parent hole) (parent i) ≤
        exchangeWeight weight hole (parent hole) i := by
    have pi := parent_lt pos
    by_cases same : i = hole
    · subst i
      simpa [exchangeWeight, (show hole ≠ parent hole by omega)] using moving
    · by_cases parentHole : parent i = hole
      · have old := ordered.bridge i valid pos parentHole positive
        simpa [exchangeWeight, other, same, parentHole, (show hole ≠ parent hole by omega)] using old
      · by_cases sibling : parent i = parent hole
        · have old := Int.le_trans moving (by simpa only [sibling] using ordered.edge i valid pos same)
          simpa [exchangeWeight, other, same, sibling] using old
        · have old := ordered.edge i valid pos same
          simpa [exchangeWeight, other, same, parentHole, sibling] using old
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro i valid pos other _
    exact newEdge i valid pos other
  · intro i valid pos child parentPositive
    have pp := parent_lt parentPositive
    have above := ordered.edge (parent hole) parentInside parentPositive (by omega)
    by_cases same : i = hole
    · subst i
      simpa [exchangeWeight, (show hole ≠ parent hole by omega), (show parent (parent hole) ≠ parent hole by omega),
        show parent (parent hole) ≠ hole by omega] using above
    · have old := Int.le_trans above (by simpa only [child] using ordered.edge i valid pos same)
      have pi := parent_lt pos
      simpa [exchangeWeight, child, same, show i ≠ parent hole by omega,
        (show parent (parent hole) ≠ parent hole by omega), show parent (parent hole) ≠ hole by omega] using old
  · intro i valid pos child
    have pi := parent_lt pos
    simpa only [child] using newEdge i valid pos (by omega)

/-- A numeric view of list entries; the default is irrelevant to in-range edges. -/
def listWeight {α : Type} (priority : α → Int) (values : List α) (i : Nat) : Int :=
  ((values[i]?).map priority).getD 0

theorem listWeight_get {α : Type} (priority : α → Int) (values : List α)
    (i : Nat) (inside : i < values.length) :
    listWeight priority values i = priority values[i] := by
  simp [listWeight, List.getElem?_eq_getElem inside]

theorem edges_iff_ordered {α : Type} (priority : α → Int) (values : List α) :
    Edges values.length (listWeight priority values) ↔ Ordered priority values := by
  constructor
  · intro ordered i inside positive
    have edge := ordered i inside positive
    rw [listWeight_get priority values i inside,
      listWeight_get priority values (parent i) (by have := parent_lt positive; omega)] at edge
    exact edge
  · intro ordered i inside positive
    rw [listWeight_get priority values i inside,
      listWeight_get priority values (parent i) (by have := parent_lt positive; omega)]
    exact ordered i inside positive

/-- Exactly the two list writes used by Python's heap swap. -/
theorem listWeight_exchange {α : Type} (priority : α → Int) (values : List α)
    (a b : Nat) (ha : a < values.length) (hb : b < values.length) :
    listWeight priority ((values.set a values[b]).set b values[a]) =
      exchangeWeight (listWeight priority values) a b := by
  funext i
  by_cases ib : i = b
  · subst i
    simp [listWeight, exchangeWeight, hb, ha]
  · by_cases ia : i = a
    · subst i
      simp [listWeight, exchangeWeight, hb, ha, ib, Ne.symm ib]
    · simp [listWeight, exchangeWeight, ib, ia, Ne.symm ib, Ne.symm ia]

/-- Appending an arbitrary entry creates an upward-repair hole at the new leaf. -/
theorem ordered_append_hole {α : Type} (priority : α → Int) (values : List α) (item : α)
    (ordered : Ordered priority values) :
    UpHole (values ++ [item]).length values.length (listWeight priority (values ++ [item])) := by
  have old (i : Nat) (inside : i < values.length) :
      listWeight priority (values ++ [item]) i = priority values[i] := by
    rw [listWeight_get priority _ i (by simp; omega), List.getElem_append_left inside]
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro i inside positive different _
    have bound : i < values.length := by simp only [List.length_append, List.length_singleton] at inside; omega
    have pi := parent_lt positive
    rw [old i bound, old (parent i) (by omega)]
    exact ordered i bound positive
  · intro i inside positive child _
    have pi := parent_lt positive
    simp only [List.length_append, List.length_singleton] at inside
    omega
  · intro i inside positive child
    have pi := parent_lt positive
    simp only [List.length_append, List.length_singleton] at inside
    omega

/-- Replacing the root does not disturb edges outside the root's children. -/
theorem ordered_replace_root_hole {α : Type} (priority : α → Int) (values : List α) (item : α)
    (ordered : Ordered priority values) :
    Hole (values.set 0 item).length 0 (listWeight priority (values.set 0 item)) := by
  constructor
  · intro i inside positive different parentDifferent
    have old := (edges_iff_ordered priority values).mpr ordered i
      (by simpa using inside) positive
    simpa [listWeight, List.getElem?_set, Ne.symm different, Ne.symm parentDifferent] using old
  · intro i inside positive child impossible
    omega

end ProjectEulerSolutions.BinaryHeapOrder
