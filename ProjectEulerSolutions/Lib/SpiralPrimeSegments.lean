import ProjectEulerSolutions.Lib.SpiralPrimes

/-! Bounded cumulative certificates with balanced counter lookup and opaque row facts. -/
namespace ProjectEulerSolutions.SpiralPrimes

/-- The endpoints overlap, allowing independently checked chunks to join at one counter. -/
structure CountSegment (num den bound first last initial final : Nat) where
  increasing : first < last
  values : Nat → Nat
  beginning : values first = initial
  ending : values last = final
  next : ∀ layer, first ≤ layer → layer < last →
    values (layer + 1) = values layer + row (layer + 1)
  earlier : ∀ layer, first < layer → layer ≤ last → layer < bound →
    num * total layer ≤ den * values layer

namespace CountSegment

variable {num den bound first middle last initial halfway final : Nat}

/-- One row supplies one atomic counter segment. The row fact remains opaque. -/
def single {increment : Nat} (checked : row (first + 1) = increment)
    (notYet : first + 1 < bound → num * total (first + 1) ≤ den * (initial + increment)) :
    CountSegment num den bound first (first + 1) initial (initial + increment) where
  increasing := Nat.lt_succ_self first
  values := fun layer => if layer ≤ first then initial else initial + increment
  beginning := by simp only [if_pos (Nat.le_refl first)]
  ending := by simp only [if_neg (show ¬ first + 1 ≤ first by omega)]
  next := by
    intro layer lower upper
    have same : layer = first := by omega
    subst layer
    simp only [if_pos (Nat.le_refl first), if_neg (show ¬ first + 1 ≤ first by omega)]
    exact (congrArg (fun value => initial + value) checked).symm
  earlier := by
    intro layer lower upper beforeBound
    have same : layer = first + 1 := by omega
    subst layer
    simp only [if_neg (show ¬ first + 1 ≤ first by omega)]
    exact notYet beforeBound

/-- Joining does not evaluate either segment's row or counter data. -/
def join (left : CountSegment num den bound first middle initial halfway)
    (right : CountSegment num den bound middle last halfway final) :
    CountSegment num den bound first last initial final where
  increasing := Nat.lt_trans left.increasing right.increasing
  values := fun layer => if layer ≤ middle then left.values layer else right.values layer
  beginning := by
    simp only [if_pos (Nat.le_of_lt left.increasing)]
    exact left.beginning
  ending := by
    simp only [if_neg (Nat.not_le_of_gt right.increasing)]
    exact right.ending
  next := by
    intro layer lower upper
    by_cases before : layer < middle
    · have nextBefore : layer + 1 ≤ middle := by omega
      simp only [if_pos (Nat.le_of_lt before), if_pos nextBefore]
      exact left.next layer lower before
    · have after : middle ≤ layer := by omega
      by_cases boundary : layer = middle
      · subst layer
        have nextRight := right.next middle (Nat.le_refl middle) right.increasing
        have sameCounter : right.values middle = left.values middle :=
          right.beginning.trans left.ending.symm
        simp only [if_pos (Nat.le_refl middle), if_neg (show ¬ middle + 1 ≤ middle by omega)]
        exact nextRight.trans (congrArg (fun count => count + row (middle + 1)) sameCounter)
      · have strictlyAfter : middle < layer := by omega
        simp only [if_neg (Nat.not_le_of_gt strictlyAfter), if_neg (show ¬ layer + 1 ≤ middle by omega)]
        exact right.next layer after upper
  earlier := by
    intro layer lower upper beforeBound
    by_cases before : layer ≤ middle
    · simp only [if_pos before]
      exact left.earlier layer lower before beforeBound
    · have after : middle < layer := by omega
      simp only [if_neg before]
      exact right.earlier layer after upper beforeBound

def asPrefix (segment : CountSegment num den bound 0 last 0 final) : CountPrefix last where
  values := segment.values
  initial := segment.beginning
  next := fun layer inside => segment.next layer (Nat.zero_le layer) inside

theorem firstCrossing (segment : CountSegment num den bound 0 bound 0 final)
    (lastCrosses : den * final < num * total bound) : FirstCrossing num den bound := by
  apply segment.asPrefix.firstCrossing
  · exact segment.increasing
  · change den * segment.values bound < num * total bound
    simpa only [segment.ending] using lastCrosses
  · intro layer positive before
    exact segment.earlier layer positive (Nat.le_of_lt before) before

end CountSegment
end ProjectEulerSolutions.SpiralPrimes
