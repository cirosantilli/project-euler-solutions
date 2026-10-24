import ProjectEulerSolutions.Lib.PolygonalFamilies

/-! Prefix indexing preserves the complete maximum-sum search over arbitrary polygonal families. -/
namespace ProjectEulerSolutions.PolygonalCycleSearch

set_option Elab.async false

abbrev Families := Int → List Int
abbrev Buckets := Int → Int → List Int

def last (path : List Int) : Int := path.getLast?.getD 0

def closed (path : List Int) : Bool :=
  match path.head?, path.getLast? with
  | some first, some final => decide (final % 100 = first / 100)
  | _, _ => false

def finish (path : List Int) : Int := if closed path then path.sum else 0

def fullScan (families : Families) : Nat → List Int → List Int → Int
  | 0, path, remaining => if remaining.isEmpty then finish path else 0
  | depth + 1, path, remaining =>
      if remaining.isEmpty then finish path
      else remaining.foldl (fun best kind =>
        let unused := remaining.filter (fun other => decide (other ≠ kind))
        (families kind).foldl (fun best value =>
          if value ∈ path then best
          else if last path % 100 = value / 100 then
            max best (fullScan families depth (path ++ [value]) unused)
          else best) best) 0

def prefixScan (buckets : Buckets) : Nat → List Int → List Int → Int
  | 0, path, remaining => if remaining.isEmpty then finish path else 0
  | depth + 1, path, remaining =>
      if remaining.isEmpty then finish path
      else remaining.foldl (fun best kind =>
        let unused := remaining.filter (fun other => decide (other ≠ kind))
        (buckets kind (last path % 100)).foldl (fun best value =>
          if value ∈ path then best
          else max best (prefixScan buckets depth (path ++ [value]) unused)) best) 0

def indexed (families : Families) : Buckets := fun kind key =>
  (families kind).filter (fun value => decide (value / 100 = key))

theorem unused_shorter (remaining : List Int) (kind : Int) (member : kind ∈ remaining) :
    (remaining.filter (fun other => decide (other ≠ kind))).length < remaining.length := by
  have bound := List.length_filter_le (fun other : Int => decide (other ≠ kind)) remaining
  have different : (remaining.filter (fun other => decide (other ≠ kind))).length ≠ remaining.length := by
    intro equal
    have kept := (List.length_filter_eq_length_iff.mp equal) kind member
    simp at kept
  omega

theorem filter_fold (items path : List Int) (visit : Int → Int) (key best : Int) :
    (items.filter (fun value => decide (value / 100 = key))).foldl
      (fun best value => if value ∈ path then best else max best (visit value)) best =
    items.foldl (fun best value => if value ∈ path then best
      else if key = value / 100 then max best (visit value) else best) best := by
  rw [List.foldl_filter]
  congr 1
  funext best value
  simp only [decide_eq_true_eq]
  by_cases used : value ∈ path
  · simp only [if_pos used, ite_self]
  · simp only [if_neg used]
    by_cases linked : value / 100 = key
    · rw [if_pos linked, if_pos linked.symm]
    · rw [if_neg linked, if_neg (Ne.symm linked)]

/-- Lookup is valid for every family, path, and remaining-type list, independently of any fixed instance. -/
theorem search_equal (families : Families) (buckets : Buckets)
    (lookup : ∀ kind key, buckets kind key = indexed families kind key)
    (depth : Nat) (path remaining : List Int) :
    prefixScan buckets depth path remaining = fullScan families depth path remaining := by
  induction depth generalizing path remaining with
  | zero => rfl
  | succ depth ih =>
      simp only [prefixScan, fullScan, ih]
      split_ifs with empty
      · rfl
      · congr 1
        funext best kind
        rw [lookup, indexed]
        exact filter_fold (families kind) path
          (fun value => fullScan families depth (path ++ [value])
            (remaining.filter (fun other => decide (other ≠ kind)))) (last path % 100) best

def remainingTypes (maxSides : Int) : List Int :=
  (List.range' 4 (maxSides - 3).toNat).map (fun (kind : Nat) => (kind : Int))

def reference (families : Families) (maxSides : Int) : Int :=
  if maxSides < 3 then 0
  else (families 3).foldl (fun best start =>
    max best (fullScan families (remainingTypes maxSides).length [start] (remainingTypes maxSides))) 0

def optimized (families : Families) (buckets : Buckets) (maxSides : Int) : Int :=
  if maxSides < 3 then 0
  else (families 3).foldl (fun best start =>
    max best (prefixScan buckets (remainingTypes maxSides).length [start] (remainingTypes maxSides))) 0

theorem models_equal (families : Families) (buckets : Buckets)
    (lookup : ∀ kind key, buckets kind key = indexed families kind key) (maxSides : Int) :
    optimized families buckets maxSides = reference families maxSides := by
  simp only [optimized, reference, search_equal families buckets lookup]

end ProjectEulerSolutions.PolygonalCycleSearch
