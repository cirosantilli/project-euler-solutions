import ProjectEulerSolutions.Lib.CliqueSearch
import Init.Data.List.Lex

/-! Soundness, minimum-sum completeness, and deterministic tie selection for
ordered finite clique search. These results quantify over every completion;
no primality assumptions or evaluation of a fixed search instance are used. -/
namespace ProjectEulerSolutions.CliqueSearch

/-- An ordered choice of the requested number of vertices, compatible at
every pair. Sublist records positions as well as membership. -/
structure Completion (relation : Nat → Nat → Bool) (need : Nat) (remaining chosen : List Nat) : Prop where
  size : chosen.length = need
  sublist : chosen.Sublist remaining
  compatible : chosen.Pairwise (fun p q => relation p q = true)

theorem Completion.enlarge {relation : Nat → Nat → Bool} {need : Nat} {small large chosen : List Nat}
    (completion : Completion relation need small chosen) (included : small.Sublist large) :
    Completion relation need large chosen :=
  ⟨completion.size, completion.sublist.trans included, completion.compatible⟩

theorem Completion.prepend {relation : Nat → Nat → Bool} {need q : Nat} {remaining chosen : List Nat}
    {sum bound : Int} (completion : Completion relation need (candidates relation (need + 1) sum q remaining bound) chosen) :
    Completion relation (need + 1) (q :: remaining) (q :: chosen) where
  size := by simp [completion.size]
  sublist := (completion.sublist.trans (candidates_sublist relation (need + 1) sum q remaining bound)).cons_cons q
  compatible := List.pairwise_cons.mpr
    ⟨fun r member => candidates_compatible relation (need + 1) sum q remaining bound r (completion.sublist.mem member),
     completion.compatible⟩

/-- A search either retains the incumbent exactly, or returns a strictly
improving compatible completion appended to the current prefix. -/
def Sound (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat) (sum : Int)
    (before after : Best) : Prop :=
  after = before ∨ ∃ chosen, Completion relation need remaining chosen ∧
    after.bound = sum + (chosen.sum : Int) ∧ after.chosen = clique ++ chosen ∧ after.bound < before.bound

theorem Sound.refl (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat) (sum : Int) (best : Best) :
    Sound relation need clique remaining sum best best := Or.inl rfl

theorem Sound.bound {relation : Nat → Nat → Bool} {need : Nat} {clique remaining : List Nat} {sum : Int}
    {before after : Best} (sound : Sound relation need clique remaining sum before after) : after.bound ≤ before.bound := by
  rcases sound with same | ⟨_, _, _, _, less⟩
  · simpa only [same] using Int.le_refl before.bound
  · omega

theorem Sound.trans {relation : Nat → Nat → Bool} {need : Nat} {clique remaining : List Nat} {sum : Int}
    {before middle after : Best} (first : Sound relation need clique remaining sum before middle)
    (second : Sound relation need clique remaining sum middle after) : Sound relation need clique remaining sum before after := by
  rcases second with same | ⟨chosen, completion, total, selected, less⟩
  · rw [same]
    exact first
  · have reduced := first.bound
    exact Or.inr ⟨chosen, completion, total, selected, by omega⟩

theorem Sound.enlarge {relation : Nat → Nat → Bool} {need : Nat} {clique small large : List Nat} {sum : Int}
    {before after : Best} (sound : Sound relation need clique small sum before after) (included : small.Sublist large) :
    Sound relation need clique large sum before after := by
  rcases sound with same | ⟨chosen, completion, total, selected, less⟩
  · exact Or.inl same
  · exact Or.inr ⟨chosen, completion.enlarge included, total, selected, less⟩

theorem Sound.prepend {relation : Nat → Nat → Bool} {need q : Nat} {clique remaining : List Nat} {sum : Int}
    {before after : Best}
    (sound : Sound relation need (clique ++ [q]) (candidates relation (need + 1) sum q remaining before.bound)
      (sum + q) before after) : Sound relation (need + 1) clique (q :: remaining) sum before after := by
  rcases sound with same | ⟨chosen, completion, total, selected, less⟩
  · exact Or.inl same
  · refine Or.inr ⟨q :: chosen, completion.prepend, ?_, ?_, less⟩
    · simpa [List.sum_cons, Int.natCast_add, Int.add_assoc] using total
    · simpa [List.append_assoc] using selected

theorem scan_sound (relation : Nat → Nat → Bool) (recurse : List Nat → List Nat → Int → Best → Best)
    (need : Nat)
    (childSound : ∀ clique remaining sum best, Sound relation need clique remaining sum best (recurse clique remaining sum best))
    (clique remaining : List Nat) (sum : Int) (best : Best) :
    Sound relation (need + 1) clique remaining sum best (scan relation recurse (need + 1) clique sum remaining best) := by
  induction remaining generalizing best with
  | nil => exact Sound.refl _ _ _ _ _ _
  | cons q rest ih =>
    rw [scan]
    split
    · exact Sound.refl _ _ _ _ _ _
    · let next := candidates relation (need + 1) sum q rest best.bound
      let updated := if need ≤ next.length then recurse (clique ++ [q]) next (sum + q) best else best
      have first : Sound relation (need + 1) clique (q :: rest) sum best updated := by
        dsimp only [updated]
        split
        · exact Sound.prepend (childSound _ _ _ _)
        · exact Sound.refl _ _ _ _ _ _
      exact first.trans ((ih updated).enlarge (List.sublist_cons_self q rest))

theorem search_sound (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat)
    (sum : Int) (best : Best) : Sound relation need clique remaining sum best (search relation need clique remaining sum best) := by
  induction need generalizing clique remaining sum best with
  | zero =>
    unfold search improve
    by_cases better : sum < best.bound
    · rw [if_pos better]
      exact Or.inr ⟨[], ⟨rfl, List.nil_sublist _, (by simp)⟩, by simp, by simp, better⟩
    · rw [if_neg better]
      exact Sound.refl _ _ _ _ _ _
  | succ need ih =>
    rw [search_succ_eq_scan]
    exact scan_sound relation (search relation need) need (fun _ _ _ _ => ih _ _ _ _) _ _ _ _

theorem bounded_prefix_eq_filter (values : List Nat) (ordered : values.Pairwise (· ≤ ·)) (base bound : Int) :
    values.takeWhile (fun (value : Nat) => decide (base + value < bound)) =
      values.filter (fun (value : Nat) => decide (base + value < bound)) := by
  induction values with
  | nil => rfl
  | cons first rest ih =>
    have sorted := List.pairwise_cons.mp ordered
    by_cases passes : base + (first : Int) < bound
    · simp only [List.takeWhile_cons, List.filter_cons, decide_eq_true_eq, passes, if_true, ih sorted.2]
    · have empty : rest.filter (fun (value : Nat) => decide (base + value < bound)) = [] := by
        apply List.filter_eq_nil_iff.mpr
        intro r member
        have lower := sorted.1 r member
        simp only [decide_eq_true_eq]
        omega
      simp only [List.takeWhile_cons, List.filter_cons, decide_eq_true_eq, passes, if_false, empty]

private theorem sublist_filter_of_all (predicate : Nat → Bool) {small large : List Nat}
    (included : small.Sublist large) (all : ∀ value ∈ small, predicate value = true) :
    small.Sublist (large.filter predicate) := by
  have same : small.filter predicate = small := List.filter_eq_self.mpr all
  simpa only [same] using included.filter predicate

theorem completion_in_candidates (relation : Nat → Nat → Bool) (need : Nat) (sum : Int) (q : Nat)
    (remaining chosen : List Nat) (bound : Int) (ordered : remaining.Pairwise (· ≤ ·))
    (completion : Completion relation need remaining chosen)
    (compatible : ∀ r ∈ chosen, relation q r = true)
    (improves : sum + (q : Int) + (chosen.sum : Int) < bound) :
    Completion relation need (candidates relation (need + 1) sum q remaining bound) chosen := by
  have members := completion_survives_candidates relation (need + 1) sum q remaining chosen bound ordered
    (fun r member => completion.sublist.mem member) compatible improves
  refine ⟨completion.size, ?_, completion.compatible⟩
  rw [candidates, bounded_prefix_eq_filter remaining ordered]
  apply sublist_filter_of_all (relation q)
  · apply sublist_filter_of_all _ completion.sublist
    intro r member
    simpa only [decide_eq_true_eq] using
      ((candidates_mem_iff relation (need + 1) sum q remaining bound ordered r).mp (members r member)).2.1
  · exact compatible

/-- Every compatible completion bounds the result of a scan from above. The
head break uses the least available vertex; candidate filtering retains all
vertices of any completion that can still improve the live incumbent. -/
theorem scan_le_completion (relation : Nat → Nat → Bool)
    (recurse : List Nat → List Nat → Int → Best → Best) (need : Nat)
    (decreases : ∀ clique remaining sum best, (recurse clique remaining sum best).bound ≤ best.bound)
    (childComplete : ∀ clique remaining sum best chosen, remaining.Pairwise (· ≤ ·) →
      Completion relation need remaining chosen → (recurse clique remaining sum best).bound ≤ sum + (chosen.sum : Int))
    (clique remaining chosen : List Nat) (sum : Int) (best : Best)
    (ordered : remaining.Pairwise (· ≤ ·)) (completion : Completion relation (need + 1) remaining chosen) :
    (scan relation recurse (need + 1) clique sum remaining best).bound ≤ sum + (chosen.sum : Int) := by
  induction remaining generalizing best with
  | nil =>
    have empty := List.sublist_nil.mp completion.sublist
    have size := completion.size
    rw [empty] at size
    simp at size
  | cons q rest ih =>
    have sorted := List.pairwise_cons.mp ordered
    by_cases cut : best.bound ≤ sum + (q : Int) * (need + 1 : Nat)
    · rw [scan, if_pos cut]
      apply bound_prunes chosen (need + 1) q sum best.bound completion.size _ cut
      intro r member
      rcases List.mem_cons.mp (completion.sublist.mem member) with same | tail
      · subst r
        exact Nat.le_refl _
      · exact sorted.1 r tail
    · rw [scan, if_neg cut]
      let next := candidates relation (need + 1) sum q rest best.bound
      let updated := if need ≤ next.length then recurse (clique ++ [q]) next (sum + q) best else best
      change (scan relation recurse (need + 1) clique sum rest updated).bound ≤ sum + (chosen.sum : Int)
      rcases List.sublist_cons_iff.mp completion.sublist with skipped | ⟨tail, same, tailSublist⟩
      · exact ih updated sorted.2 ⟨completion.size, skipped, completion.compatible⟩
      · subst chosen
        have tailCompletion : Completion relation need rest tail := by
          refine ⟨?_, tailSublist, (List.pairwise_cons.mp completion.compatible).2⟩
          have size := completion.size
          simp only [List.length_cons] at size
          omega
        have finalBound := scan_bound relation recurse decreases (need + 1) clique sum rest updated
        by_cases already : best.bound ≤ sum + ((q :: tail).sum : Int)
        · have updatedBound : updated.bound ≤ best.bound := by
            dsimp only [updated]
            split
            · exact decreases _ _ _ _
            · exact Int.le_refl _
          omega
        · have improves : sum + (q : Int) + (tail.sum : Int) < best.bound := by
            simp only [List.sum_cons, Int.natCast_add] at already
            omega
          have nextCompletion : Completion relation need next tail := completion_in_candidates relation need sum q rest tail
            best.bound sorted.2 tailCompletion (List.pairwise_cons.mp completion.compatible).1 improves
          have enough : need ≤ next.length := by
            have size := nextCompletion.sublist.length_le
            rw [nextCompletion.size] at size
            exact size
          have sortedNext : next.Pairwise (· ≤ ·) := List.Pairwise.sublist (candidates_sublist _ _ _ _ _ _) sorted.2
          have childBound := childComplete (clique ++ [q]) next (sum + q) best tail sortedNext nextCompletion
          have updatedBound : updated.bound ≤ sum + (q : Int) + (tail.sum : Int) := by
            simpa only [updated, if_pos enough] using childBound
          simp only [List.sum_cons, Int.natCast_add]
          omega

theorem search_le_completion (relation : Nat → Nat → Bool) (need : Nat) (clique remaining chosen : List Nat)
    (sum : Int) (best : Best) (ordered : remaining.Pairwise (· ≤ ·))
    (completion : Completion relation need remaining chosen) :
    (search relation need clique remaining sum best).bound ≤ sum + (chosen.sum : Int) := by
  induction need generalizing clique remaining chosen sum best with
  | zero =>
    have empty : chosen = [] := List.length_eq_zero_iff.mp completion.size
    subst chosen
    simp only [search, improve]
    split <;> simp_all
  | succ need ih =>
    rw [search_succ_eq_scan]
    exact scan_le_completion relation (search relation need) need (search_bound relation need)
      (fun _ _ _ _ _ ordered completion => ih _ _ _ _ _ ordered completion) _ _ _ _ _ ordered completion

/-- The original incumbent survives exactly when no compatible completion
has a smaller total; this includes the no-completion case. -/
theorem search_eq_incumbent_iff (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat)
    (sum : Int) (best : Best) (ordered : remaining.Pairwise (· ≤ ·)) :
    search relation need clique remaining sum best = best ↔
      ∀ chosen, Completion relation need remaining chosen → best.bound ≤ sum + (chosen.sum : Int) := by
  constructor
  · intro same chosen completion
    have minimal := search_le_completion relation need clique remaining chosen sum best ordered completion
    simpa only [same] using minimal
  · intro optimal
    rcases search_sound relation need clique remaining sum best with same | ⟨chosen, completion, total, _, improves⟩
    · exact same
    · have lower := optimal chosen completion
      omega

def Precedes (left right : List Nat) : Prop := left = right ∨ List.Lex (· < ·) left right

theorem Precedes.antisymm {left right : List Nat} (first : Precedes left right) (second : Precedes right left) : left = right := by
  rcases first with same | less
  · exact same
  · rcases second with same | greater
    · exact same.symm
    · exact False.elim (List.lex_asymm Nat.lt_asymm less greater)

private theorem lex_append_prefix (front : List Nat) {left right : List Nat}
    (less : List.Lex (· < ·) left right) : List.Lex (· < ·) (front ++ left) (front ++ right) := by
  induction front with
  | nil => exact less
  | cons head rest ih => exact List.Lex.cons ih

/-- Among improved results with the same sum, retain the first increasing
completion. An existing incumbent wins ties regardless of its selected list. -/
def First (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat) (sum : Int)
    (before after : Best) : Prop :=
  ∀ chosen, Completion relation need remaining chosen → sum + (chosen.sum : Int) = after.bound →
    after.bound < before.bound → Precedes after.chosen (clique ++ chosen)

private theorem child_precedes_later (relation : Nat → Nat → Bool) (need : Nat)
    (clique rest chosen : List Nat) (q : Nat) (sum : Int) (before after : Best)
    (sound : Sound relation need (clique ++ [q]) rest sum before after) (improves : after.bound < before.bound)
    (later : List Nat) (ordered : (q :: later).Pairwise (· < ·))
    (completion : Completion relation (need + 1) later chosen) : Precedes after.chosen (clique ++ chosen) := by
  rcases sound with same | ⟨extension, _, _, selected, _⟩
  · rw [same] at improves
    omega
  · cases chosen with
    | nil => have size := completion.size; simp at size
    | cons head tail =>
      have less := (List.pairwise_cons.mp ordered).1 head (completion.sublist.mem (by simp))
      apply Or.inr
      rw [selected, List.append_assoc]
      exact lex_append_prefix clique (List.Lex.rel less)

theorem scan_first (relation : Nat → Nat → Bool)
    (recurse : List Nat → List Nat → Int → Best → Best) (need : Nat)
    (decreases : ∀ clique remaining sum best, (recurse clique remaining sum best).bound ≤ best.bound)
    (stable : ∀ clique remaining sum best,
      (recurse clique remaining sum best).bound = best.bound → recurse clique remaining sum best = best)
    (childSound : ∀ clique remaining sum best, Sound relation need clique remaining sum best (recurse clique remaining sum best))
    (childComplete : ∀ clique remaining sum best chosen, remaining.Pairwise (· ≤ ·) →
      Completion relation need remaining chosen → (recurse clique remaining sum best).bound ≤ sum + (chosen.sum : Int))
    (childFirst : ∀ clique remaining sum best, remaining.Pairwise (· < ·) →
      First relation need clique remaining sum best (recurse clique remaining sum best))
    (clique remaining : List Nat) (sum : Int) (best : Best) (ordered : remaining.Pairwise (· < ·)) :
    First relation (need + 1) clique remaining sum best (scan relation recurse (need + 1) clique sum remaining best) := by
  induction remaining generalizing best with
  | nil =>
    intro chosen completion total improves
    change best.bound < best.bound at improves
    omega
  | cons q rest ih =>
    have sorted := List.pairwise_cons.mp ordered
    by_cases cut : best.bound ≤ sum + (q : Int) * (need + 1 : Nat)
    · rw [scan, if_pos cut]
      intro chosen completion total improves
      omega
    · rw [scan, if_neg cut]
      let next := candidates relation (need + 1) sum q rest best.bound
      let updated := if need ≤ next.length then recurse (clique ++ [q]) next (sum + q) best else best
      change First relation (need + 1) clique (q :: rest) sum best (scan relation recurse (need + 1) clique sum rest updated)
      intro chosen completion total improves
      have finalBound := scan_bound relation recurse decreases (need + 1) clique sum rest updated
      rcases List.sublist_cons_iff.mp completion.sublist with skipped | ⟨tail, same, tailSublist⟩
      · have skippedCompletion : Completion relation (need + 1) rest chosen := ⟨completion.size, skipped, completion.compatible⟩
        by_cases improvedAgain : (scan relation recurse (need + 1) clique sum rest updated).bound < updated.bound
        · exact ih updated sorted.2 chosen skippedCompletion total improvedAgain
        · have unchanged := scan_equal_bound relation recurse decreases stable (need + 1) clique sum rest updated (by omega)
          have improvedChild : updated.bound < best.bound := by omega
          by_cases enough : need ≤ next.length
          · have childImproves : (recurse (clique ++ [q]) next (sum + q) best).bound < best.bound := by
              simpa only [updated, if_pos enough] using improvedChild
            rw [unchanged]
            simpa only [updated, if_pos enough] using child_precedes_later relation need clique next chosen q (sum + q) best
              (recurse (clique ++ [q]) next (sum + q) best) (childSound _ _ _ _) childImproves rest ordered skippedCompletion
          · simp only [updated, if_neg enough] at improvedChild
            omega
      · subst chosen
        have tailCompletion : Completion relation need rest tail := by
          refine ⟨?_, tailSublist, (List.pairwise_cons.mp completion.compatible).2⟩
          have size := completion.size
          simp only [List.length_cons] at size
          omega
        have improvesCompletion : sum + (q : Int) + (tail.sum : Int) < best.bound := by
          simp only [List.sum_cons, Int.natCast_add] at total
          omega
        have nondecreasing : rest.Pairwise (· ≤ ·) := sorted.2.imp (fun less => Nat.le_of_lt less)
        have nextCompletion : Completion relation need next tail := completion_in_candidates relation need sum q rest tail
          best.bound nondecreasing tailCompletion (List.pairwise_cons.mp completion.compatible).1 improvesCompletion
        have enough : need ≤ next.length := by
          have size := nextCompletion.sublist.length_le
          rw [nextCompletion.size] at size
          exact size
        have sortedNext : next.Pairwise (· < ·) := List.Pairwise.sublist (candidates_sublist _ _ _ _ _ _) sorted.2
        have childBound := childComplete (clique ++ [q]) next (sum + q) best tail
          (sortedNext.imp (fun less => Nat.le_of_lt less)) nextCompletion
        have updatedBound : updated.bound ≤ sum + (q : Int) + (tail.sum : Int) := by
          simpa only [updated, if_pos enough] using childBound
        have equal : (scan relation recurse (need + 1) clique sum rest updated).bound = updated.bound := by
          simp only [List.sum_cons, Int.natCast_add] at total
          omega
        have unchanged := scan_equal_bound relation recurse decreases stable (need + 1) clique sum rest updated equal
        have updatedValue : updated.bound = (recurse (clique ++ [q]) next (sum + q) best).bound := by
          simp only [updated, if_pos enough]
        have childEqual : sum + (q : Int) + (tail.sum : Int) = (recurse (clique ++ [q]) next (sum + q) best).bound := by
          simp only [List.sum_cons, Int.natCast_add] at total
          omega
        have childImproves : (recurse (clique ++ [q]) next (sum + q) best).bound < best.bound := by
          omega
        have earliest := childFirst (clique ++ [q]) next (sum + q) best sortedNext tail nextCompletion childEqual childImproves
        rw [unchanged]
        simpa only [updated, if_pos enough, List.append_assoc, List.singleton_append] using earliest

theorem search_first (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat)
    (sum : Int) (best : Best) (ordered : remaining.Pairwise (· < ·)) :
    First relation need clique remaining sum best (search relation need clique remaining sum best) := by
  induction need generalizing clique remaining sum best with
  | zero =>
    intro chosen completion total improves
    have empty : chosen = [] := List.length_eq_zero_iff.mp completion.size
    subst chosen
    by_cases better : sum < best.bound
    · simp only [search, improve, if_pos better, List.append_nil]
      exact Or.inl rfl
    · simp only [search, improve, if_neg better] at improves
      omega
  | succ need ih =>
    rw [search_succ_eq_scan]
    exact scan_first relation (search relation need) need (search_bound relation need) (search_equal_bound relation need)
      (search_sound relation need)
      (fun clique remaining sum best chosen sorted completion => search_le_completion relation need clique remaining chosen sum best sorted completion)
      (fun _ _ _ _ sorted => ih _ _ _ _ sorted) _ _ _ _ ordered

/-- A complete result specification: retain the incumbent or select a valid
strict improvement, attain a minimum, and retain the earliest equal minimum. -/
structure Optimal (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat) (sum : Int)
    (before after : Best) : Prop where
  sound : Sound relation need clique remaining sum before after
  minimal : ∀ chosen, Completion relation need remaining chosen → after.bound ≤ sum + (chosen.sum : Int)
  first : First relation need clique remaining sum before after

theorem search_optimal (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat)
    (sum : Int) (best : Best) (ordered : remaining.Pairwise (· < ·)) :
    Optimal relation need clique remaining sum best (search relation need clique remaining sum best) :=
  ⟨search_sound relation need clique remaining sum best,
   fun chosen completion => search_le_completion relation need clique remaining chosen sum best
     (ordered.imp (fun less => Nat.le_of_lt less)) completion,
   search_first relation need clique remaining sum best ordered⟩

/-- The specification determines both the numerical answer and the selected
list, including the sentinel and pre-existing-incumbent cases. -/
theorem Optimal.unique {relation : Nat → Nat → Bool} {need : Nat} {clique remaining : List Nat} {sum : Int}
    {before left right : Best} (first : Optimal relation need clique remaining sum before left)
    (second : Optimal relation need clique remaining sum before right) : left = right := by
  rcases first.sound with unchanged | ⟨chosenLeft, completionLeft, totalLeft, selectedLeft, improvesLeft⟩
  · rcases second.sound with same | ⟨chosenRight, completionRight, totalRight, _, improvesRight⟩
    · exact unchanged.trans same.symm
    · have lower := first.minimal chosenRight completionRight
      rw [unchanged] at lower
      omega
  · rcases second.sound with unchanged | ⟨chosenRight, completionRight, totalRight, selectedRight, improvesRight⟩
    · have lower := second.minimal chosenLeft completionLeft
      rw [unchanged] at lower
      omega
    · have leftBound := first.minimal chosenRight completionRight
      have rightBound := second.minimal chosenLeft completionLeft
      have sameBound : left.bound = right.bound := by omega
      have leftFirst : Precedes left.chosen right.chosen := by
        rw [selectedRight]
        exact first.first chosenRight completionRight (by omega) improvesLeft
      have rightFirst : Precedes right.chosen left.chosen := by
        rw [selectedLeft]
        exact second.first chosenLeft completionLeft (by omega) improvesRight
      have sameChosen := leftFirst.antisymm rightFirst
      cases left
      cases right
      exact congrArg₂ Best.mk sameBound sameChosen

theorem search_eq_of_optimal (relation : Nat → Nat → Bool) (need : Nat) (clique remaining : List Nat)
    (sum : Int) (best result : Best) (ordered : remaining.Pairwise (· < ·))
    (specification : Optimal relation need clique remaining sum best result) :
    search relation need clique remaining sum best = result :=
  (search_optimal relation need clique remaining sum best ordered).unique specification

end ProjectEulerSolutions.CliqueSearch
