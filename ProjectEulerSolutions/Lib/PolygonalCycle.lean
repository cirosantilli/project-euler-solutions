import Mathlib.Data.List.Basic

/-! Ordered search models for the two Python implementations of Euler 61. -/
namespace ProjectEulerSolutions.PolygonalCycle

universe u v
variable {α : Type u} {β : Type v}

abbrev Node := Int × Int
abbrev Path := List Node

def polygonal (sides index : Int) : Int :=
  index * ((sides - 2) * index - (sides - 4)) / 2

def fourDigit (limit sides : Int) : List Int :=
  ((List.range (limit + 1).toNat).map (fun index => polygonal sides index)).filter
    (fun value => decide (1000 ≤ value ∧ value ≤ 9999))

def candidates (sides : Int) : List Int :=
  ((List.range 140).map (fun offset : Nat => polygonal sides (1 + offset))).filter
    (fun value => decide (1000 ≤ value ∧ value ≤ 9999 ∧ 10 ≤ value % 100))

def types : List Int := [3, 4, 5, 6, 7, 8]

def links (left right : Int) : Bool := decide (left % 100 = right / 100)

def closed (path : List Int) : Bool :=
  match path.head?, path.getLast? with
  | some first, some last => links last first
  | _, _ => false

/-- Ordered, short-circuiting search agrees with the source's early returns. -/
def firstResult (values : List α) (visit : α → Option β) : Option β :=
  match values with
  | [] => none
  | value :: rest =>
      match visit value with
      | some result => some result
      | none => firstResult rest visit

def finish (path : Path) : Option Path :=
  if closed (path.map Prod.fst) then some path else none

def search : Nat → Path → Option Path
  | 0, path => if path.length = 6 then finish path else none
  | fuel + 1, path =>
      if path.length = 6 then finish path
      else
        let suffix := ((path.getLast?).getD (0, 0)).1 % 100
        firstResult types fun kind =>
          if kind ∈ path.map Prod.snd then none
          else
            firstResult ((candidates kind).filter (fun value => decide (value / 100 = suffix))) fun value =>
              if value ∈ path.map Prod.fst then none
              else search fuel (path ++ [(value, kind)])

def cycle : Option Path :=
  firstResult (candidates 8) (fun start => search 5 [(start, 8)])

def answer : Int := ((cycle.map (fun path => (path.map Prod.fst).sum)).getD 0)

def extend (limit : Int) : Nat → List Int → List Nat → Int
  | 0, path, remaining =>
      if remaining.isEmpty && closed path then path.sum else 0
  | fuel + 1, path, remaining =>
      if remaining.isEmpty then if closed path then path.sum else 0
      else
        remaining.foldl (fun best kind =>
          let unused := remaining.filter (fun other => decide (other ≠ kind))
          (fourDigit limit (3 + kind)).foldl (fun best value =>
            if value ∈ path then best
            else if links ((path.getLast?).getD 0) value then
              max best (extend limit fuel (path ++ [value]) unused)
            else best) best) 0

def reference (limit : Int) : Int :=
  (fourDigit limit 3).foldl (fun best start => max best (extend limit 5 [start] [1, 2, 3, 4, 5])) 0

end ProjectEulerSolutions.PolygonalCycle
