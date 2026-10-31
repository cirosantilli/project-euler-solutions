import ProjectEulerSolutions.Lib.NonnegativeWalkBounds

/-! Synchronous relaxation for nonnegative vertex-weighted graphs.
Distances include the costs of both endpoints. The sentinel is an input bound,
and each finite estimate is witnessed by an actual walk. -/
namespace ProjectEulerSolutions.VertexRelaxation

variable {α : Type}

def step (cost : α → Int) (adjacent : α → List α) (distance : α → Int) (v : α) : Int :=
  (adjacent v).foldl (fun best u => min best (cost v + distance u)) (distance v)

def initial [DecidableEq α] (cost : α → Int) (infinity : Int) (source v : α) : Int :=
  if v = source then cost source else infinity

def run [DecidableEq α] (cost : α → Int) (adjacent : α → List α)
    (infinity : Int) (source : α) : Nat → α → Int
  | 0 => initial cost infinity source
  | n + 1 => step cost adjacent (run cost adjacent infinity source n)

private theorem fold_le_initial (values : List α) (f : α → Int) (best : Int) :
    values.foldl (fun acc u => min acc (f u)) best ≤ best := by
  induction values generalizing best with
  | nil => exact le_refl _
  | cons first rest ih =>
      exact (ih (min best (f first))).trans (Int.min_le_left _ _)

private theorem fold_le_member (values : List α) (f : α → Int) (best : Int)
    (u : α) (member : u ∈ values) :
    values.foldl (fun acc v => min acc (f v)) best ≤ f u := by
  induction values generalizing best with
  | nil => simp at member
  | cons first rest ih =>
      rcases List.mem_cons.mp member with rfl | member
      · exact (fold_le_initial rest f _).trans (Int.min_le_right _ _)
      · exact ih _ member

theorem step_le (cost : α → Int) (adjacent : α → List α) (distance : α → Int) (v : α) :
    step cost adjacent distance v ≤ distance v :=
  fold_le_initial _ _ _

theorem step_le_neighbor (cost : α → Int) (adjacent : α → List α)
    (distance : α → Int) (v u : α) (member : u ∈ adjacent v) :
    step cost adjacent distance v ≤ cost v + distance u :=
  fold_le_member _ _ _ _ member

private theorem add_fold_min (values : List α) (f : α → Int) (best amount : Int) :
    amount + values.foldl (fun acc u => min acc (f u)) best =
      values.foldl (fun acc u => min acc (amount + f u)) (amount + best) := by
  induction values generalizing best with
  | nil => rfl
  | cons first rest ih =>
      simp only [List.foldl_cons]
      rw [ih]
      congr 1
      omega

private theorem min_fold_min (values : List α) (f : α → Int) (best outer : Int) :
    min outer (values.foldl (fun acc u => min acc (f u)) best) =
      values.foldl (fun acc u => min acc (f u)) (min outer best) := by
  induction values generalizing best with
  | nil => rfl
  | cons first rest ih =>
      simp only [List.foldl_cons]
      rw [ih]
      congr 1
      omega

/-- Including the current estimate among the neighbors is harmless for nonnegative costs. -/
theorem step_eq_cost_fold (cost : α → Int) (adjacent : α → List α)
    (distance : α → Int) (v : α) (nonnegative : 0 ≤ cost v) :
    step cost adjacent distance v = min (distance v)
      (cost v + (adjacent v).foldl (fun acc u => min acc (distance u)) (distance v)) := by
  rw [add_fold_min, min_fold_min]
  rw [min_eq_left (show distance v ≤ cost v + distance v by omega)]
  rfl

theorem run_antitone [DecidableEq α] (cost : α → Int) (adjacent : α → List α)
    (infinity : Int) (source v : α) : Antitone (fun n => run cost adjacent infinity source n v) := by
  apply antitone_nat_of_succ_le
  intro n
  exact step_le _ _ _ _

/-- After enough rounds every walk provides an upper bound on the estimate. -/
theorem run_le_walk [DecidableEq α] (graph : SimpleGraph α) (cost : α → Int)
    (adjacent : α → List α) (complete : ∀ v u, graph.Adj v u → u ∈ adjacent v)
    (infinity : Int) {source v : α} (walk : graph.Walk v source) (n : Nat)
    (enough : walk.length ≤ n) :
    run cost adjacent infinity source n v ≤ (walk.support.map cost).sum := by
  induction walk generalizing n with
  | @nil u =>
      have bound := run_antitone cost adjacent infinity u u (Nat.zero_le n)
      simpa [run, initial] using bound
  | @cons v u source edge tail ih =>
      cases n with
      | zero => simp at enough
      | succ n =>
          have rest := ih n (by simpa using enough)
          have next := step_le_neighbor cost adjacent (run cost adjacent infinity source n)
            v u (complete v u edge)
          simp only [SimpleGraph.Walk.support_cons, List.map_cons, List.sum_cons]
          exact next.trans (Int.add_le_add_left rest _)

def Witness (graph : SimpleGraph α) (cost : α → Int) (infinity : Int)
    (source v : α) (value : Int) : Prop :=
  value ≤ infinity ∧ (value = infinity ∨
    ∃ walk : graph.Walk v source, (walk.support.map cost).sum = value)

private theorem fold_witness (graph : SimpleGraph α) (cost : α → Int)
    (infinity : Int) (source v : α) (distance : α → Int)
    (nonnegative : 0 ≤ cost v)
    (known : ∀ u, Witness graph cost infinity source u (distance u))
    (values : List α) (edges : ∀ u ∈ values, graph.Adj v u)
    (best : Int) (witness : Witness graph cost infinity source v best) :
    Witness graph cost infinity source v
      (values.foldl (fun acc u => min acc (cost v + distance u)) best) := by
  induction values generalizing best with
  | nil => exact witness
  | cons u rest ih =>
      apply ih (fun u member => edges u (by simp [member]))
      dsimp only
      by_cases smaller : cost v + distance u < best
      · rw [min_eq_right (le_of_lt smaller)]
        refine ⟨by have bound := witness.1; omega, ?_⟩
        rcases (known u).2 with unreachable | ⟨walk, value⟩
        · have bound := witness.1
          omega
        · right
          refine ⟨.cons (edges u (by simp)) walk, ?_⟩
          simp [value]
      · rw [min_eq_left (by omega)]
        exact witness

/-- The sentinel never gives rise to a spurious finite path. -/
theorem run_witness [DecidableEq α] (graph : SimpleGraph α) (cost : α → Int)
    (adjacent : α → List α) (edges : ∀ v u, u ∈ adjacent v → graph.Adj v u)
    (nonnegative : ∀ v, 0 ≤ cost v) (infinity : Int) (source : α)
    (sourceBound : cost source ≤ infinity) (n : Nat) :
    ∀ v, Witness graph cost infinity source v (run cost adjacent infinity source n v) := by
  induction n with
  | zero =>
      intro v
      by_cases same : v = source
      · subst v
        refine ⟨by simpa [run, initial] using sourceBound, Or.inr ⟨.nil, ?_⟩⟩
        simp [run, initial]
      · simp [run, initial, same, Witness]
  | succ n ih =>
      intro v
      exact fold_witness graph cost infinity source v _ (nonnegative v) ih
        (adjacent v) (edges v) _ (ih v)

/-- The vertex count suffices: erasing cycles produces a cheaper path with fewer edges.
The enumeration need not have any particular order. -/
theorem run_minimum_enumerated [DecidableEq α] (vertices : List α)
    (complete : ∀ vertex, vertex ∈ vertices) (graph : SimpleGraph α) (cost : α → Int)
    (adjacent : α → List α)
    (adjacency : ∀ v u, u ∈ adjacent v ↔ graph.Adj v u)
    (nonnegative : ∀ vertex, 0 ≤ cost vertex)
    (source v : α) (reachable : graph.Walk v source) :
    ∃ shortest : graph.Walk v source,
      (shortest.support.map cost).sum =
        run cost adjacent ((vertices.map cost).sum + 1) source vertices.length v ∧
      ∀ walk : graph.Walk v source,
        (shortest.support.map cost).sum ≤ (walk.support.map cost).sum := by
  have bounded (walk : graph.Walk v source) :
      run cost adjacent ((vertices.map cost).sum + 1) source vertices.length v ≤
        (walk.support.map cost).sum := by
    obtain ⟨path, _, cheaper, short, _⟩ :=
      NonnegativeWalkBounds.bounded_path_enumerated vertices complete graph cost nonnegative walk
    exact (run_le_walk graph cost adjacent (fun v u edge => (adjacency v u).mpr edge)
      _ path _ (Nat.le_of_lt short)).trans cheaper
  obtain ⟨path, _, _, _, below⟩ :=
    NonnegativeWalkBounds.bounded_path_enumerated vertices complete graph cost nonnegative reachable
  have strictlyBelow := lt_of_le_of_lt (bounded path) below
  have sourceBound : cost source ≤ (vertices.map cost).sum + 1 := by
    have bound := NonnegativeWalkBounds.cost_le_of_nodup_subset cost [source] vertices
      (by simp) (by simpa using complete source) (fun vertex _ => nonnegative vertex)
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero] at bound
    omega
  have witness := run_witness graph cost adjacent
    (fun v u member => (adjacency v u).mp member) nonnegative _ source sourceBound vertices.length v
  rcases witness.2 with sentinel | ⟨shortest, value⟩
  · omega
  · exact ⟨shortest, value, fun walk => value ▸ bounded walk⟩

/-- Flat-index arrays are a specialization of the enumeration theorem. -/
theorem run_minimum (weights : List Int) (graph : SimpleGraph (Fin weights.length))
    (adjacent : Fin weights.length → List (Fin weights.length))
    (adjacency : ∀ v u, u ∈ adjacent v ↔ graph.Adj v u)
    (nonnegative : ∀ value ∈ weights, 0 ≤ value)
    (source v : Fin weights.length) (reachable : graph.Walk v source) :
    ∃ shortest : graph.Walk v source,
      (shortest.support.map weights.get).sum =
        run weights.get adjacent (weights.sum + 1) source weights.length v ∧
      ∀ walk : graph.Walk v source,
        (shortest.support.map weights.get).sum ≤ (walk.support.map weights.get).sum := by
  simpa only [List.length_finRange, List.map_get_finRange] using
    run_minimum_enumerated (List.finRange weights.length) (by intro vertex; simp)
      graph weights.get adjacent adjacency
      (fun vertex => nonnegative _ (List.get_mem weights vertex)) source v reachable

/-- Changing vertex representation preserves the recurrence, including neighbor order. -/
theorem run_map {β : Type} [DecidableEq α] [DecidableEq β]
    (encode : α → β) (injective : Function.Injective encode)
    (costA : α → Int) (costB : β → Int) (costs : ∀ v, costA v = costB (encode v))
    (adjacentA : α → List α) (adjacentB : β → List β)
    (neighbors : ∀ v, (adjacentA v).map encode = adjacentB (encode v))
    (infinity : Int) (source : α) (n : Nat) :
    ∀ v, run costA adjacentA infinity source n v =
      run costB adjacentB infinity (encode source) n (encode v) := by
  induction n with
  | zero =>
      intro v
      simp only [run, initial, injective.eq_iff, costs]
  | succ n ih =>
      intro v
      simp only [run, step, ← neighbors, List.foldl_map, costs, ih]

end ProjectEulerSolutions.VertexRelaxation
