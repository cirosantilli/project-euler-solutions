import ProjectEulerSolutions.Lib.AlmostEquilateral

/-! Agreement of the complete bounded reference enumeration and the increasing Pell traversal. -/
namespace ProjectEulerSolutions.AlmostEquilateral
set_option Elab.async false

noncomputable def emitted (limit a : Int) : List Int := by
  classical
  exact (if Triangle a (a - 1) ∧ 3 * a - 1 ≤ limit then [3 * a - 1] else []) ++
    (if Triangle a (a + 1) ∧ 3 * a + 1 ≤ limit then [3 * a + 1] else [])

noncomputable def reference (limit : Int) : List Int :=
  (List.range' 2 (((limit + 1) / 3 - 1).toNat)).flatMap fun (a : Nat) => emitted limit (a : Int)

theorem mem_emitted (limit a p : Int) : p ∈ emitted limit a ↔
    (Triangle a (a - 1) ∧ 3 * a - 1 ≤ limit ∧ p = 3 * a - 1) ∨
    (Triangle a (a + 1) ∧ 3 * a + 1 ≤ limit ∧ p = 3 * a + 1) := by
  classical
  unfold emitted
  split_ifs <;> simp_all

theorem emitted_nodup (limit a : Int) : (emitted limit a).Nodup := by
  classical
  unfold emitted
  split_ifs <;> simp [show 3 * a - 1 ≠ 3 * a + 1 by omega]

theorem emitted_disjoint (limit : Int) (a b : Nat) (different : a ≠ b) :
    List.Disjoint (emitted limit a) (emitted limit b) := by
  apply List.disjoint_left.mpr
  intro p first second
  rw [mem_emitted] at first second
  rcases first with first | first <;> rcases second with second | second <;> omega

theorem reference_nodup (limit : Int) : (reference limit).Nodup := by
  apply List.nodup_flatMap.mpr
  refine ⟨fun _ _ => emitted_nodup _ _, ?_⟩
  exact (List.nodup_range').imp fun different => emitted_disjoint limit _ _ different

theorem reference_mem (limit p : Int) : p ∈ reference limit ↔
    ∃ a b : Int, Triangle a b ∧ 2 * a + b ≤ limit ∧ p = 2 * a + b := by
  constructor
  · intro member
    obtain ⟨a, _, member⟩ := List.mem_flatMap.mp member
    rcases (mem_emitted limit a p).mp member with ⟨valid, bounded, same⟩ | ⟨valid, bounded, same⟩
    · exact ⟨a, a - 1, valid, by omega, by omega⟩
    · exact ⟨a, a + 1, valid, by omega, by omega⟩
  · rintro ⟨a, b, valid, bounded, same⟩
    have positive := valid.1
    have base := valid.2.1
    have lower : (a.toNat : Int) = a := Int.toNat_of_nonneg (by omega)
    apply List.mem_flatMap.mpr
    refine ⟨a.toNat, ?_, ?_⟩
    · rw [List.mem_range']
      refine ⟨a.toNat - 2, ?_, by omega⟩
      rcases base with base | base <;> omega
    · rw [mem_emitted, lower]
      rcases base with rfl | rfl
      · exact Or.inr ⟨valid, by omega, by omega⟩
      · exact Or.inl ⟨valid, by omega, by omega⟩

theorem perimeter_strictMono_tail : StrictMono fun n => perimeter (n + 2) := by
  apply strictMono_nat_of_lt_succ
  intro n
  exact perimeter_increasing (n + 2) (by omega)

theorem perimeter_lt {n k : Nat} (nontrivial : 2 ≤ n) (later : n < k) : perimeter n < perimeter k := by
  have compared := perimeter_strictMono_tail (show n - 2 < k - 2 by omega)
  simpa only [Nat.sub_add_cancel nontrivial, Nat.sub_add_cancel (show 2 ≤ k by omega)] using compared

theorem perimeter_le {n k : Nat} (nontrivial : 2 ≤ n) (later : n ≤ k) : perimeter n ≤ perimeter k := by
  rcases later.eq_or_lt with rfl | later
  · rfl
  · exact (perimeter_lt nontrivial later).le

theorem perimetersFrom_mem (limit : Int) (n : Nat) (nontrivial : 2 ≤ n) (p : Int) :
    p ∈ perimetersFrom limit n nontrivial ↔
      ∃ k, n ≤ k ∧ perimeter k ≤ limit ∧ p = perimeter k := by
  rw [perimetersFrom]
  by_cases done : limit < perimeter n
  · rw [if_pos done]
    simp only [List.not_mem_nil, false_iff]
    rintro ⟨k, later, bounded, _⟩
    have := perimeter_le nontrivial later
    omega
  · rw [if_neg done, List.mem_cons, perimetersFrom_mem limit (n + 1) (by omega)]
    constructor
    · rintro (same | ⟨k, later, bounded, same⟩)
      · exact ⟨n, le_rfl, by omega, same⟩
      · exact ⟨k, by omega, bounded, same⟩
    · rintro ⟨k, later, bounded, same⟩
      by_cases equal : k = n
      · exact Or.inl (by simpa [equal] using same)
      · exact Or.inr ⟨k, by omega, bounded, same⟩
termination_by (limit + 3 - 2 * (solution n).x).toNat
decreasing_by
  have bounds := perimeter_bounds n
  have growth : (solution n).x < (solution (n + 1)).x := solution_x_strictMono (Nat.lt_succ_self n)
  omega

theorem perimetersFrom_nodup (limit : Int) (n : Nat) (nontrivial : 2 ≤ n) :
    (perimetersFrom limit n nontrivial).Nodup := by
  rw [perimetersFrom]
  split_ifs with done
  · exact List.nodup_nil
  · apply List.nodup_cons.mpr
    constructor
    · intro member
      obtain ⟨k, later, _, same⟩ := (perimetersFrom_mem limit (n + 1) (by omega) _).mp member
      have := perimeter_lt nontrivial (show n < k by omega)
      omega
    · exact perimetersFrom_nodup limit (n + 1) (by omega)
termination_by (limit + 3 - 2 * (solution n).x).toNat
decreasing_by
  have bounds := perimeter_bounds n
  have growth : (solution n).x < (solution (n + 1)).x := solution_x_strictMono (Nat.lt_succ_self n)
  omega

/-- Both enumerations visit every admissible perimeter exactly once. -/
theorem reference_perm (limit : Int) :
    (reference limit).Perm (perimetersFrom limit 2 (by decide)) := by
  apply (List.perm_ext_iff_of_nodup (reference_nodup limit) (perimetersFrom_nodup limit 2 (by decide))).mpr
  intro p
  rw [reference_mem, perimetersFrom_mem]
  constructor
  · rintro ⟨a, b, valid, bounded, same⟩
    obtain ⟨n, nontrivial, pair⟩ := triangle_complete a b valid
    exact ⟨n, nontrivial, by simpa [perimeter, pair] using bounded, by simpa [perimeter, pair] using same⟩
  · rintro ⟨n, nontrivial, bounded, same⟩
    exact ⟨(triangle n).1, (triangle n).2, triangle_valid n nontrivial, bounded, same⟩

theorem reference_sum (limit : Int) :
    (reference limit).sum = (perimetersFrom limit 2 (by decide)).sum := (reference_perm limit).sum_eq

end ProjectEulerSolutions.AlmostEquilateral
