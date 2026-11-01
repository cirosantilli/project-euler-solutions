import ProjectEulerSolutions.Lib.DigitSquareReachability

/-! Exact classification caches for complete digit-square walks and path compression. -/
namespace ProjectEulerSolutions.DigitSquareChains.Memo
open ProjectEulerStatements.BoundedOrbit
set_option Elab.async false

abbrev Cache := Nat → Option Nat

structure Valid (known : Cache) : Prop where
  target : known 89 = some 89
  sound : ∀ n value, known n = some value → decide (value = 89) = endsAt89 n

def initial : Cache := fun n => if n = 0 then some 0 else if n = 1 then some 1 else if n = 89 then some 89 else none

theorem initial_valid : Valid initial := by
  constructor
  · simp [initial]
  · intro n value found
    by_cases zero : n = 0
    · subst n; simp [initial] at found; subst value; simp
    · by_cases one : n = 1
      · subst n; simp [initial] at found; subst value; simp
      · by_cases target : n = 89
        · subst n; simp [initial] at found; subst value; simp
        · simp [initial, zero, one, target] at found

def write (known : Cache) (key value : Nat) : Cache := fun n => if n = key then some value else known n

theorem Valid.write {known : Cache} (valid : Valid known) (key value : Nat)
    (correct : decide (value = 89) = endsAt89 key) : Valid (write known key value) := by
  constructor
  · by_cases target : key = 89
    · subst key
      have same : value = 89 := by simpa only [endsAt89_target, decide_eq_true_eq] using correct
      simp [Memo.write, same]
    · simpa [Memo.write, Ne.symm target] using valid.target
  · intro n result found
    by_cases same : n = key
    · subst n
      have equal : value = result := by simpa [Memo.write] using found
      simpa only [equal] using correct
    · apply valid.sound n result
      simpa [Memo.write, same] using found

def fill (known : Cache) (path : List Nat) (value : Nat) : Cache :=
  path.foldl (fun cache n => write cache n value) known

theorem fill_valid (known : Cache) (path : List Nat) (value : Nat) (valid : Valid known)
    (correct : ∀ n ∈ path, decide (value = 89) = endsAt89 n) : Valid (fill known path value) := by
  induction path generalizing known with
  | nil => exact valid
  | cons head tail ih =>
      exact ih (write known head value) (valid.write head value (correct head (by simp)))
        (by intro n member; exact correct n (by simp [member]))

/-- Cached endpoints and repeated values stop the walk; every other value is recorded before advancing. -/
inductive Trace (known : Cache) : Nat → List Nat → Nat → List Nat → Nat → Prop
  | cached {current path value} : known current = some value → Trace known current path current path value
  | repeated {current path} : known current = none → current ∈ path → Trace known current path current path 0
  | step {current path last result value} : known current = none → current ∉ path →
      Trace known (nextTerm current) (path ++ [current]) last result value →
      Trace known current path last result value

theorem exists_trace (known : Cache) (start current : Nat) (path : List Nat)
    (bounded : current ≤ bound start) (distinct : path.Nodup) (inside : Within (bound start) path) :
    ∃ last result value, Trace known current path last result value := by
  cases found : known current with
  | some value => exact ⟨current, path, value, .cached found⟩
  | none =>
      by_cases repeated : current ∈ path
      · exact ⟨current, path, 0, .repeated found repeated⟩
      · obtain ⟨last, result, value, rest⟩ := exists_trace known start (nextTerm current) (path ++ [current])
          (nextTerm_bounded start current bounded) (append_distinct path current distinct repeated)
          (append_within (bound start) path current inside bounded)
        exact ⟨last, result, value, .step found repeated rest⟩
termination_by bound start + 1 - path.length
decreasing_by
  have enough := room (bound start) current path distinct inside bounded repeated
  simp only [List.length_append, List.length_singleton]
  omega

/-- Every preceding value has the same classification as the current frontier. -/
def Uniform (current : Nat) (path : List Nat) : Prop :=
  ∀ n ∈ path, endsAt89 n = endsAt89 current

theorem Uniform.advance {current : Nat} {path : List Nat} (uniform : Uniform current path) :
    Uniform (nextTerm current) (path ++ [current]) := by
  intro n member
  rw [endsAt89_next]
  rcases List.mem_append.mp member with earlier | last
  · exact uniform n earlier
  · simp only [List.mem_singleton.mp last]

/-- The returned label and every path entry agree with exact target reachability. -/
theorem Trace.correct {known : Cache} {current last value : Nat} {path result : List Nat}
    (trace : Trace known current path last result value) (valid : Valid known)
    (frontier : Frontier current path) (uniform : Uniform current path) :
    decide (value = 89) = endsAt89 current ∧
      ∀ n ∈ result, decide (value = 89) = endsAt89 n := by
  induction trace with
  | cached found =>
      have classified := valid.sound _ _ found
      exact ⟨classified, fun n member => classified.trans (uniform n member).symm⟩
  | repeated _ present =>
      have classified : decide ((0 : Nat) = 89) = endsAt89 _ := (frontier.repeated present).symm
      exact ⟨classified, fun n member => classified.trans (uniform n member).symm⟩
  | @step current path last result value missing fresh trace ih =>
      have notTarget : current ≠ 89 := by
        intro same
        subst current
        simp [valid.target] at missing
      obtain ⟨classified, paths⟩ := ih (frontier.advance notTarget) uniform.advance
      exact ⟨classified.trans (endsAt89_next current), paths⟩

theorem Trace.fill_valid {known : Cache} {current last value : Nat} {path result : List Nat}
    (trace : Trace known current path last result value) (valid : Valid known)
    (frontier : Frontier current path) (uniform : Uniform current path) :
    Valid (fill known result value) :=
  Memo.fill_valid known result value valid (trace.correct valid frontier uniform).2

/-- Every start admits a complete classified walk and a sound compressed cache, without a search budget. -/
theorem from_start (known : Cache) (valid : Valid known) (start : Nat) :
    ∃ last path value, Trace known start [] last path value ∧
      decide (value = 89) = endsAt89 start ∧ Valid (fill known path value) := by
  obtain ⟨last, path, value, trace⟩ := exists_trace known start start []
    (Nat.le_max_left _ _) (by simp) (by simp [Within])
  have frontier := Frontier.empty start
  have uniform : Uniform start [] := by intro n member; cases member
  exact ⟨last, path, value, trace, (trace.correct valid frontier uniform).1,
    trace.fill_valid valid frontier uniform⟩

end ProjectEulerSolutions.DigitSquareChains.Memo
