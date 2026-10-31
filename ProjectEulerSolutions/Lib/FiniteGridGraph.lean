import ProjectEulerSolutions.Lib.VertexRelaxation
import ProjectEulerStatements.P83

/-! Finite four-direction grids, with the neighbor order used by the Python reference. -/
namespace ProjectEulerSolutions.FiniteGridGraph

abbrev Vertex (height width : Nat) := Fin height × Fin width
abbrev Position := Int × Int

def encode {height width : Nat} (v : Vertex height width) : Position := (v.1.val, v.2.val)

theorem encode_injective (height width : Nat) :
    Function.Injective (@encode height width) := by
  intro v u equal
  have rows := congrArg Prod.fst equal
  have cols := congrArg Prod.snd equal
  apply Prod.ext <;> apply Fin.ext <;> simp only [encode, Int.natCast_inj] at * <;> assumption

def near (p q : Position) : Prop :=
  (p.1 = q.1 ∧ (p.2 + 1 = q.2 ∨ q.2 + 1 = p.2)) ∨
    (p.2 = q.2 ∧ (p.1 + 1 = q.1 ∨ q.1 + 1 = p.1))

def graph (height width : Nat) : SimpleGraph (Vertex height width) where
  Adj v u := near (encode v) (encode u)
  symm := by intro v u edge; unfold near at *; omega
  loopless := ⟨by intro v; unfold near; omega⟩

theorem neighbor_bounds (height width : Nat) (p q : Position)
    (member : q ∈ ProjectEulerStatements.P83.neighbors p.1 p.2 height width) :
    0 ≤ q.1 ∧ 0 ≤ q.2 ∧ q.1 < height ∧ q.2 < width := by
  simpa [ProjectEulerStatements.P83.neighbors] using (List.mem_filter.mp member).2

def decode {height width : Nat} (p : Position)
    (inside : 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 < height ∧ p.2 < width) : Vertex height width :=
  (⟨p.1.toNat, by omega⟩, ⟨p.2.toNat, by omega⟩)

theorem encode_decode {height width : Nat} (p : Position)
    (inside : 0 ≤ p.1 ∧ 0 ≤ p.2 ∧ p.1 < height ∧ p.2 < width) :
    encode (decode p inside) = p := by
  ext <;> simp [encode, decode, Int.toNat_of_nonneg inside.1, Int.toNat_of_nonneg inside.2.1]

def adjacent {height width : Nat} (v : Vertex height width) : List (Vertex height width) :=
  (ProjectEulerStatements.P83.neighbors (encode v).1 (encode v).2 height width).attach.map
    (fun q => decode q.val (neighbor_bounds height width (encode v) q.val q.property))

theorem adjacent_encode {height width : Nat} (v : Vertex height width) :
    (adjacent v).map encode =
      ProjectEulerStatements.P83.neighbors (encode v).1 (encode v).2 height width := by
  simp only [adjacent, List.map_map, Function.comp_def, encode_decode, List.attach_map_subtype_val]

theorem adjacent_iff {height width : Nat} (v u : Vertex height width) :
    u ∈ adjacent v ↔ (graph height width).Adj v u := by
  have member : u ∈ adjacent v ↔ encode u ∈ (adjacent v).map encode :=
    (List.mem_map_of_injective (encode_injective height width)).symm
  rw [member, adjacent_encode]
  have rowBound := u.1.isLt
  have colBound := u.2.isLt
  simp only [ProjectEulerStatements.P83.neighbors, List.mem_filter, List.mem_cons,
    List.not_mem_nil, or_false, decide_eq_true_eq, graph, near, encode, Prod.mk.injEq]
  omega

def vertices (height width : Nat) : List (Vertex height width) :=
  (List.finRange height).flatMap fun r => (List.finRange width).map fun c => (r, c)

theorem mem_vertices {height width : Nat} (v : Vertex height width) :
    v ∈ vertices height width := by
  simp [vertices]

theorem vertices_length (height width : Nat) :
    (vertices height width).length = height * width := by
  simp [vertices, List.length_flatMap]

def origin {height width : Nat} (rowsPositive : 0 < height) (colsPositive : 0 < width) :
    Vertex height width := (⟨0, rowsPositive⟩, ⟨0, colsPositive⟩)

/-- Moving toward row zero and then column zero constructs a path for every cell. -/
theorem reachable_origin {height width : Nat} (rowsPositive : 0 < height) (colsPositive : 0 < width)
    (v : Vertex height width) :
    Nonempty ((graph height width).Walk v (origin rowsPositive colsPositive)) := by
  by_cases rowZero : v.1.val = 0
  · by_cases colZero : v.2.val = 0
    · have same : v = origin rowsPositive colsPositive :=
        Prod.ext (Fin.ext rowZero) (Fin.ext colZero)
      subst v
      exact ⟨.nil⟩
    · let previous : Vertex height width := (v.1, ⟨v.2.val - 1, by have := v.2.isLt; omega⟩)
      obtain ⟨walk⟩ := reachable_origin rowsPositive colsPositive previous
      refine ⟨.cons ?_ walk⟩
      change near (encode v) (encode previous)
      simp only [near, encode, previous]
      rw [Int.natCast_sub (show 1 ≤ v.2.val by omega)]
      exact Or.inl ⟨trivial, Or.inr (by omega)⟩
  · let previous : Vertex height width := (⟨v.1.val - 1, by have := v.1.isLt; omega⟩, v.2)
    obtain ⟨walk⟩ := reachable_origin rowsPositive colsPositive previous
    refine ⟨.cons ?_ walk⟩
    change near (encode v) (encode previous)
    simp only [near, encode, previous]
    rw [Int.natCast_sub (show 1 ≤ v.1.val by omega)]
    exact Or.inr ⟨trivial, Or.inr (by omega)⟩
termination_by v.1.val + v.2.val
decreasing_by all_goals omega

end ProjectEulerSolutions.FiniteGridGraph
