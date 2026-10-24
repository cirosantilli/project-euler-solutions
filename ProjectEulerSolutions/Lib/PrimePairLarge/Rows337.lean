import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11159_small : List Nat := []
theorem row11159_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11159 row11159_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11159_block_005 : List Nat := []
theorem row11159_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11159 row11159_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11159_block_006 : List Nat := [11633, 12227]
def plan11159_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 11), .possible, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .possible, (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 37), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11159_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11159 row11159_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11159_block_006
  · decide
  · rfl

def row11159_block_007 : List Nat := [12437, 12659, 12923, 13859, 14447]
def plan11159_block_007 : List Budget.Choice := [.first, .first, .possible, .first, .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 17), (.firstFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .possible, (.secondFactor 19), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 13), (.firstFactor 7), .first, .possible, (.firstFactor 11), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 37), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 13), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .second, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .second, .first, .first, (.secondFactor 13), .first, (.secondFactor 19), (.secondFactor 23), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .possible, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 37), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 13), .first, .first, .possible, .first, .first]
set_option maxRecDepth 2048 in
theorem row11159_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11159 row11159_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11159_block_007
  · decide
  · rfl

def row11159_block_008 : List Nat := [14717]
def plan11159_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 29), .first, (.secondFactor 23), (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 23), (.secondFactor 7), .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 19), .first, (.secondFactor 31), .first, .possible, (.firstFactor 7), .first, .first, .second, .second, (.secondFactor 7), (.firstFactor 23), .first, .second, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11159_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11159 row11159_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11159_block_008
  · decide
  · rfl

def row11159_block_009 : List Nat := []
theorem row11159_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11159 row11159_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11159_block_010 : List Nat := []
theorem row11159_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11159 row11159_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11159_block_011 : List Nat := []
theorem row11159_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11159 row11159_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11159_block_012 : List Nat := []
theorem row11159_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11159 row11159_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11159_block_013 : List Nat := []
theorem row11159_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11159 row11159_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11159_large_left_left : List Nat := row11159_small ++ row11159_block_005
theorem row11159_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11159 row11159_large_left_left :=
  Budget.RowUpperBound.branch row11159_small_covers row11159_block_005_covers

def row11159_large_left_right_right : List Nat := row11159_block_007 ++ row11159_block_008
theorem row11159_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11159 row11159_large_left_right_right :=
  Budget.RowUpperBound.branch row11159_block_007_covers row11159_block_008_covers

def row11159_large_left_right : List Nat := row11159_block_006 ++ row11159_large_left_right_right
theorem row11159_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11159 row11159_large_left_right :=
  Budget.RowUpperBound.branch row11159_block_006_covers row11159_large_left_right_right_covers

def row11159_large_left : List Nat := row11159_large_left_left ++ row11159_large_left_right
theorem row11159_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11159 row11159_large_left :=
  Budget.RowUpperBound.branch row11159_large_left_left_covers row11159_large_left_right_covers

def row11159_large_right_left : List Nat := row11159_block_009 ++ row11159_block_010
theorem row11159_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11159 row11159_large_right_left :=
  Budget.RowUpperBound.branch row11159_block_009_covers row11159_block_010_covers

def row11159_large_right_right_right : List Nat := row11159_block_012 ++ row11159_block_013
theorem row11159_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11159 row11159_large_right_right_right :=
  Budget.RowUpperBound.branch row11159_block_012_covers row11159_block_013_covers

def row11159_large_right_right : List Nat := row11159_block_011 ++ row11159_large_right_right_right
theorem row11159_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11159 row11159_large_right_right :=
  Budget.RowUpperBound.branch row11159_block_011_covers row11159_large_right_right_right_covers

def row11159_large_right : List Nat := row11159_large_right_left ++ row11159_large_right_right
theorem row11159_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11159 row11159_large_right :=
  Budget.RowUpperBound.branch row11159_large_right_left_covers row11159_large_right_right_covers

def row11159_large : List Nat := row11159_large_left ++ row11159_large_right
theorem row11159_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11159 row11159_large :=
  Budget.RowUpperBound.branch row11159_large_left_covers row11159_large_right_covers

def row11159 : List Nat := row11159_large
theorem row11159_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11159 row11159 := row11159_large_covers

def row11161_small : List Nat := []
theorem row11161_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11161 row11161_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11161_block_005 : List Nat := []
theorem row11161_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11161 row11161_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11161_block_006 : List Nat := []
def plan11161_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 19), .first, (.firstFactor 17), .first, .first, .first, .first, .second, .second, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 17), (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.secondFactor 19), .first, (.secondFactor 17), (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 29), .first, (.secondFactor 29), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11161_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11161 row11161_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11161_block_006
  · decide
  · rfl

def row11161_block_007 : List Nat := [12799, 13309, 13693, 14071, 14437]
def plan11161_block_007 : List Budget.Choice := [(.firstFactor 19), (.secondFactor 23), .first, (.secondFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 37), (.firstFactor 13), .first, .first, .second, (.firstFactor 11), (.secondFactor 13), .first, (.firstFactor 17), .second, (.firstFactor 29), (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 19), (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 23), .first, .first, .first, .second, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, (.secondFactor 17), (.firstFactor 17), (.firstFactor 13), .possible, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 7), .first, .second, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .possible, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 13), .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), .possible, .first, (.secondFactor 37), (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11161_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11161 row11161_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11161_block_007
  · decide
  · rfl

def row11161_block_008 : List Nat := []
def plan11161_block_008 : List Budget.Choice := [.second, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .second, .first, .first, (.firstFactor 29), .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11161_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11161 row11161_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11161_block_008
  · decide
  · rfl

def row11161_block_009 : List Nat := []
theorem row11161_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11161 row11161_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11161_block_010 : List Nat := []
theorem row11161_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11161 row11161_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11161_block_011 : List Nat := []
theorem row11161_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11161 row11161_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11161_block_012 : List Nat := []
theorem row11161_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11161 row11161_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11161_block_013 : List Nat := []
theorem row11161_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11161 row11161_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11161_large_left_left : List Nat := row11161_small ++ row11161_block_005
theorem row11161_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11161 row11161_large_left_left :=
  Budget.RowUpperBound.branch row11161_small_covers row11161_block_005_covers

def row11161_large_left_right_right : List Nat := row11161_block_007 ++ row11161_block_008
theorem row11161_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11161 row11161_large_left_right_right :=
  Budget.RowUpperBound.branch row11161_block_007_covers row11161_block_008_covers

def row11161_large_left_right : List Nat := row11161_block_006 ++ row11161_large_left_right_right
theorem row11161_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11161 row11161_large_left_right :=
  Budget.RowUpperBound.branch row11161_block_006_covers row11161_large_left_right_right_covers

def row11161_large_left : List Nat := row11161_large_left_left ++ row11161_large_left_right
theorem row11161_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11161 row11161_large_left :=
  Budget.RowUpperBound.branch row11161_large_left_left_covers row11161_large_left_right_covers

def row11161_large_right_left : List Nat := row11161_block_009 ++ row11161_block_010
theorem row11161_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11161 row11161_large_right_left :=
  Budget.RowUpperBound.branch row11161_block_009_covers row11161_block_010_covers

def row11161_large_right_right_right : List Nat := row11161_block_012 ++ row11161_block_013
theorem row11161_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11161 row11161_large_right_right_right :=
  Budget.RowUpperBound.branch row11161_block_012_covers row11161_block_013_covers

def row11161_large_right_right : List Nat := row11161_block_011 ++ row11161_large_right_right_right
theorem row11161_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11161 row11161_large_right_right :=
  Budget.RowUpperBound.branch row11161_block_011_covers row11161_large_right_right_right_covers

def row11161_large_right : List Nat := row11161_large_right_left ++ row11161_large_right_right
theorem row11161_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11161 row11161_large_right :=
  Budget.RowUpperBound.branch row11161_large_right_left_covers row11161_large_right_right_covers

def row11161_large : List Nat := row11161_large_left ++ row11161_large_right
theorem row11161_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11161 row11161_large :=
  Budget.RowUpperBound.branch row11161_large_left_covers row11161_large_right_covers

def row11161 : List Nat := row11161_large
theorem row11161_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11161 row11161 := row11161_large_covers

def row11171_small : List Nat := []
theorem row11171_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11171 row11171_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11171_block_005 : List Nat := []
theorem row11171_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11171 row11171_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11171_block_006 : List Nat := [11279, 11321, 12413]
def plan11171_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .possible, .first, .first, .first, .first, .possible, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 13), .second, .first, .first, (.secondFactor 29), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, .second, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 37), (.firstFactor 29), .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 17), .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 37), .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 37), .first, (.secondFactor 23), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 13), .first, .possible]
set_option maxRecDepth 2048 in
theorem row11171_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11171 row11171_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11171_block_006
  · decide
  · rfl

def row11171_block_007 : List Nat := [13967, 14051]
def plan11171_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 31), (.firstFactor 13), .first, .first, .first, (.secondFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 19), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, (.secondFactor 23), (.firstFactor 31), .first, (.secondFactor 13), .first, .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 17), .first, .second, .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 29), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), (.secondFactor 37), .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 23), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .possible, (.secondFactor 23), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11171_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11171 row11171_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11171_block_007
  · decide
  · rfl

def row11171_block_008 : List Nat := []
def plan11171_block_008 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 31), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.secondFactor 29), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11171_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11171 row11171_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11171_block_008
  · decide
  · rfl

def row11171_block_009 : List Nat := []
theorem row11171_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11171 row11171_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11171_block_010 : List Nat := []
theorem row11171_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11171 row11171_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11171_block_011 : List Nat := []
theorem row11171_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11171 row11171_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11171_block_012 : List Nat := []
theorem row11171_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11171 row11171_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11171_block_013 : List Nat := []
theorem row11171_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11171 row11171_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11171_large_left_left : List Nat := row11171_small ++ row11171_block_005
theorem row11171_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11171 row11171_large_left_left :=
  Budget.RowUpperBound.branch row11171_small_covers row11171_block_005_covers

def row11171_large_left_right_right : List Nat := row11171_block_007 ++ row11171_block_008
theorem row11171_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11171 row11171_large_left_right_right :=
  Budget.RowUpperBound.branch row11171_block_007_covers row11171_block_008_covers

def row11171_large_left_right : List Nat := row11171_block_006 ++ row11171_large_left_right_right
theorem row11171_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11171 row11171_large_left_right :=
  Budget.RowUpperBound.branch row11171_block_006_covers row11171_large_left_right_right_covers

def row11171_large_left : List Nat := row11171_large_left_left ++ row11171_large_left_right
theorem row11171_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11171 row11171_large_left :=
  Budget.RowUpperBound.branch row11171_large_left_left_covers row11171_large_left_right_covers

def row11171_large_right_left : List Nat := row11171_block_009 ++ row11171_block_010
theorem row11171_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11171 row11171_large_right_left :=
  Budget.RowUpperBound.branch row11171_block_009_covers row11171_block_010_covers

def row11171_large_right_right_right : List Nat := row11171_block_012 ++ row11171_block_013
theorem row11171_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11171 row11171_large_right_right_right :=
  Budget.RowUpperBound.branch row11171_block_012_covers row11171_block_013_covers

def row11171_large_right_right : List Nat := row11171_block_011 ++ row11171_large_right_right_right
theorem row11171_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11171 row11171_large_right_right :=
  Budget.RowUpperBound.branch row11171_block_011_covers row11171_large_right_right_right_covers

def row11171_large_right : List Nat := row11171_large_right_left ++ row11171_large_right_right
theorem row11171_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11171 row11171_large_right :=
  Budget.RowUpperBound.branch row11171_large_right_left_covers row11171_large_right_right_covers

def row11171_large : List Nat := row11171_large_left ++ row11171_large_right
theorem row11171_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11171 row11171_large :=
  Budget.RowUpperBound.branch row11171_large_left_covers row11171_large_right_covers

def row11171 : List Nat := row11171_large
theorem row11171_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11171 row11171 := row11171_large_covers

def row11173_small : List Nat := []
theorem row11173_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11173 row11173_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11173_block_005 : List Nat := []
theorem row11173_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11173 row11173_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11173_block_006 : List Nat := [11719]
def plan11173_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 37), .first, (.secondFactor 31), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .possible, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .second, (.firstFactor 31), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .second, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .second, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11173_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11173 row11173_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11173_block_006
  · decide
  · rfl

def row11173_block_007 : List Nat := [12739, 12967, 13063, 13789]
def plan11173_block_007 : List Budget.Choice := [(.firstFactor 37), (.secondFactor 13), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .possible, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .possible, (.firstFactor 7), (.secondFactor 13), .first, .first, (.secondFactor 7), .first, .second, .second, .first, .first, .first, .possible, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .second, .first, (.firstFactor 23), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.secondFactor 37), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), .first, (.firstFactor 29), .first, .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .second, .first, (.firstFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 31), (.firstFactor 19), .first, .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row11173_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11173 row11173_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11173_block_007
  · decide
  · rfl

def row11173_block_008 : List Nat := [14797]
def plan11173_block_008 : List Budget.Choice := [.first, .first, (.secondFactor 19), .first, .second, .first, .first, .first, (.firstFactor 19), (.secondFactor 7), .first, .second, .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11173_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11173 row11173_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11173_block_008
  · decide
  · rfl

def row11173_block_009 : List Nat := []
theorem row11173_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11173 row11173_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11173_block_010 : List Nat := []
theorem row11173_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11173 row11173_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11173_block_011 : List Nat := []
theorem row11173_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11173 row11173_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11173_block_012 : List Nat := []
theorem row11173_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11173 row11173_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11173_block_013 : List Nat := []
theorem row11173_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11173 row11173_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11173_large_left_left : List Nat := row11173_small ++ row11173_block_005
theorem row11173_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11173 row11173_large_left_left :=
  Budget.RowUpperBound.branch row11173_small_covers row11173_block_005_covers

def row11173_large_left_right_right : List Nat := row11173_block_007 ++ row11173_block_008
theorem row11173_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11173 row11173_large_left_right_right :=
  Budget.RowUpperBound.branch row11173_block_007_covers row11173_block_008_covers

def row11173_large_left_right : List Nat := row11173_block_006 ++ row11173_large_left_right_right
theorem row11173_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11173 row11173_large_left_right :=
  Budget.RowUpperBound.branch row11173_block_006_covers row11173_large_left_right_right_covers

def row11173_large_left : List Nat := row11173_large_left_left ++ row11173_large_left_right
theorem row11173_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11173 row11173_large_left :=
  Budget.RowUpperBound.branch row11173_large_left_left_covers row11173_large_left_right_covers

def row11173_large_right_left : List Nat := row11173_block_009 ++ row11173_block_010
theorem row11173_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11173 row11173_large_right_left :=
  Budget.RowUpperBound.branch row11173_block_009_covers row11173_block_010_covers

def row11173_large_right_right_right : List Nat := row11173_block_012 ++ row11173_block_013
theorem row11173_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11173 row11173_large_right_right_right :=
  Budget.RowUpperBound.branch row11173_block_012_covers row11173_block_013_covers

def row11173_large_right_right : List Nat := row11173_block_011 ++ row11173_large_right_right_right
theorem row11173_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11173 row11173_large_right_right :=
  Budget.RowUpperBound.branch row11173_block_011_covers row11173_large_right_right_right_covers

def row11173_large_right : List Nat := row11173_large_right_left ++ row11173_large_right_right
theorem row11173_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11173 row11173_large_right :=
  Budget.RowUpperBound.branch row11173_large_right_left_covers row11173_large_right_right_covers

def row11173_large : List Nat := row11173_large_left ++ row11173_large_right
theorem row11173_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11173 row11173_large :=
  Budget.RowUpperBound.branch row11173_large_left_covers row11173_large_right_covers

def row11173 : List Nat := row11173_large
theorem row11173_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11173 row11173 := row11173_large_covers

end ProjectEulerSolutions.PrimePairLarge
