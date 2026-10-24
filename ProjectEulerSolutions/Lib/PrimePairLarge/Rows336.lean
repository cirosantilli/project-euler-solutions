import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11117_small : List Nat := []
theorem row11117_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11117 row11117_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11117_block_005 : List Nat := []
theorem row11117_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11117 row11117_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11117_block_006 : List Nat := [12347]
def plan11117_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 29), .first, .second, .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), .second, (.secondFactor 37), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .second, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 17), .first, (.secondFactor 19), (.secondFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, (.firstFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 29), (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 31), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 37), (.firstFactor 11), .first, .first, (.firstFactor 7), .second, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .possible, .first, .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11117_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11117 row11117_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11117_block_006
  · decide
  · rfl

def row11117_block_007 : List Nat := [13121, 13523, 13709, 14369]
def plan11117_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.secondFactor 23), (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 13), (.secondFactor 19), (.secondFactor 7), (.firstFactor 23), (.firstFactor 17), .second, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 29), (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 29), (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, .first, (.firstFactor 31), (.firstFactor 13), .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 23), (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 13), .first, .possible, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .second, .possible, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, .first, .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 13), .first, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11117_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11117 row11117_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11117_block_007
  · decide
  · rfl

def row11117_block_008 : List Nat := []
def plan11117_block_008 : List Budget.Choice := [.first, .first, .first, (.secondFactor 37), .first, (.secondFactor 31), (.firstFactor 37), (.firstFactor 11), .first, .first, (.secondFactor 23), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, (.secondFactor 19), .first, .second, (.firstFactor 7), .first, .first, (.secondFactor 37), (.firstFactor 11), (.secondFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11117_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11117 row11117_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11117_block_008
  · decide
  · rfl

def row11117_block_009 : List Nat := []
theorem row11117_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11117 row11117_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11117_block_010 : List Nat := []
theorem row11117_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11117 row11117_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11117_block_011 : List Nat := []
theorem row11117_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11117 row11117_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11117_block_012 : List Nat := []
theorem row11117_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11117 row11117_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11117_block_013 : List Nat := []
theorem row11117_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11117 row11117_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11117_large_left_left : List Nat := row11117_small ++ row11117_block_005
theorem row11117_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11117 row11117_large_left_left :=
  Budget.RowUpperBound.branch row11117_small_covers row11117_block_005_covers

def row11117_large_left_right_right : List Nat := row11117_block_007 ++ row11117_block_008
theorem row11117_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11117 row11117_large_left_right_right :=
  Budget.RowUpperBound.branch row11117_block_007_covers row11117_block_008_covers

def row11117_large_left_right : List Nat := row11117_block_006 ++ row11117_large_left_right_right
theorem row11117_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11117 row11117_large_left_right :=
  Budget.RowUpperBound.branch row11117_block_006_covers row11117_large_left_right_right_covers

def row11117_large_left : List Nat := row11117_large_left_left ++ row11117_large_left_right
theorem row11117_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11117 row11117_large_left :=
  Budget.RowUpperBound.branch row11117_large_left_left_covers row11117_large_left_right_covers

def row11117_large_right_left : List Nat := row11117_block_009 ++ row11117_block_010
theorem row11117_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11117 row11117_large_right_left :=
  Budget.RowUpperBound.branch row11117_block_009_covers row11117_block_010_covers

def row11117_large_right_right_right : List Nat := row11117_block_012 ++ row11117_block_013
theorem row11117_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11117 row11117_large_right_right_right :=
  Budget.RowUpperBound.branch row11117_block_012_covers row11117_block_013_covers

def row11117_large_right_right : List Nat := row11117_block_011 ++ row11117_large_right_right_right
theorem row11117_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11117 row11117_large_right_right :=
  Budget.RowUpperBound.branch row11117_block_011_covers row11117_large_right_right_right_covers

def row11117_large_right : List Nat := row11117_large_right_left ++ row11117_large_right_right
theorem row11117_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11117 row11117_large_right :=
  Budget.RowUpperBound.branch row11117_large_right_left_covers row11117_large_right_right_covers

def row11117_large : List Nat := row11117_large_left ++ row11117_large_right
theorem row11117_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11117 row11117_large :=
  Budget.RowUpperBound.branch row11117_large_left_covers row11117_large_right_covers

def row11117 : List Nat := row11117_large
theorem row11117_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11117 row11117 := row11117_large_covers

def row11119_small : List Nat := []
theorem row11119_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11119 row11119_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11119_block_005 : List Nat := []
theorem row11119_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11119 row11119_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11119_block_006 : List Nat := [11467]
def plan11119_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 31), (.firstFactor 23), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .possible, .first, .first, .first, (.secondFactor 13), (.secondFactor 29), (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 23), (.firstFactor 19), .first, .first, .first, .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 7), .second, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 37), .first, (.secondFactor 7), .second, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row11119_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11119 row11119_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11119_block_006
  · decide
  · rfl

def row11119_block_007 : List Nat := [12589, 12757, 13477]
def plan11119_block_007 : List Budget.Choice := [(.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 13), (.secondFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 19), .first, .first, (.firstFactor 29), (.secondFactor 13), .possible, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .second, (.firstFactor 7), .first, .possible, (.firstFactor 13), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 7), (.firstFactor 17), (.firstFactor 13), (.secondFactor 31), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, (.secondFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 29), (.secondFactor 37), (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 23), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.secondFactor 7), (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 37), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 19), .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 13), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11119_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11119 row11119_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11119_block_007
  · decide
  · rfl

def row11119_block_008 : List Nat := []
def plan11119_block_008 : List Budget.Choice := [(.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 19), .first, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11119_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11119 row11119_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11119_block_008
  · decide
  · rfl

def row11119_block_009 : List Nat := []
theorem row11119_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11119 row11119_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11119_block_010 : List Nat := []
theorem row11119_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11119 row11119_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11119_block_011 : List Nat := []
theorem row11119_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11119 row11119_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11119_block_012 : List Nat := []
theorem row11119_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11119 row11119_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11119_block_013 : List Nat := []
theorem row11119_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11119 row11119_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11119_large_left_left : List Nat := row11119_small ++ row11119_block_005
theorem row11119_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11119 row11119_large_left_left :=
  Budget.RowUpperBound.branch row11119_small_covers row11119_block_005_covers

def row11119_large_left_right_right : List Nat := row11119_block_007 ++ row11119_block_008
theorem row11119_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11119 row11119_large_left_right_right :=
  Budget.RowUpperBound.branch row11119_block_007_covers row11119_block_008_covers

def row11119_large_left_right : List Nat := row11119_block_006 ++ row11119_large_left_right_right
theorem row11119_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11119 row11119_large_left_right :=
  Budget.RowUpperBound.branch row11119_block_006_covers row11119_large_left_right_right_covers

def row11119_large_left : List Nat := row11119_large_left_left ++ row11119_large_left_right
theorem row11119_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11119 row11119_large_left :=
  Budget.RowUpperBound.branch row11119_large_left_left_covers row11119_large_left_right_covers

def row11119_large_right_left : List Nat := row11119_block_009 ++ row11119_block_010
theorem row11119_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11119 row11119_large_right_left :=
  Budget.RowUpperBound.branch row11119_block_009_covers row11119_block_010_covers

def row11119_large_right_right_right : List Nat := row11119_block_012 ++ row11119_block_013
theorem row11119_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11119 row11119_large_right_right_right :=
  Budget.RowUpperBound.branch row11119_block_012_covers row11119_block_013_covers

def row11119_large_right_right : List Nat := row11119_block_011 ++ row11119_large_right_right_right
theorem row11119_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11119 row11119_large_right_right :=
  Budget.RowUpperBound.branch row11119_block_011_covers row11119_large_right_right_right_covers

def row11119_large_right : List Nat := row11119_large_right_left ++ row11119_large_right_right
theorem row11119_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11119 row11119_large_right :=
  Budget.RowUpperBound.branch row11119_large_right_left_covers row11119_large_right_right_covers

def row11119_large : List Nat := row11119_large_left ++ row11119_large_right
theorem row11119_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11119 row11119_large :=
  Budget.RowUpperBound.branch row11119_large_left_covers row11119_large_right_covers

def row11119 : List Nat := row11119_large
theorem row11119_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11119 row11119 := row11119_large_covers

def row11131_small : List Nat := []
theorem row11131_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11131 row11131_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11131_block_005 : List Nat := []
theorem row11131_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11131 row11131_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11131_block_006 : List Nat := []
def plan11131_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 23), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 13), (.secondFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 31), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 29), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, (.secondFactor 23), .first, (.secondFactor 29), .first]
set_option maxRecDepth 2048 in
theorem row11131_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11131 row11131_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11131_block_006
  · decide
  · rfl

def row11131_block_007 : List Nat := [13999]
def plan11131_block_007 : List Budget.Choice := [(.secondFactor 31), .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 23), (.firstFactor 11), .first, .second, .first, .second, (.secondFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 29), .second, (.firstFactor 19), (.firstFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 37), (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 37), .first, (.firstFactor 19), .first, (.secondFactor 19), .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .possible, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.secondFactor 19), (.secondFactor 31), .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 19), (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row11131_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11131 row11131_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11131_block_007
  · decide
  · rfl

def row11131_block_008 : List Nat := []
def plan11131_block_008 : List Budget.Choice := [(.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .second, .first, (.secondFactor 17), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11131_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11131 row11131_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11131_block_008
  · decide
  · rfl

def row11131_block_009 : List Nat := []
theorem row11131_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11131 row11131_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11131_block_010 : List Nat := []
theorem row11131_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11131 row11131_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11131_block_011 : List Nat := []
theorem row11131_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11131 row11131_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11131_block_012 : List Nat := []
theorem row11131_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11131 row11131_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11131_block_013 : List Nat := []
theorem row11131_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11131 row11131_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11131_large_left_left : List Nat := row11131_small ++ row11131_block_005
theorem row11131_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11131 row11131_large_left_left :=
  Budget.RowUpperBound.branch row11131_small_covers row11131_block_005_covers

def row11131_large_left_right_right : List Nat := row11131_block_007 ++ row11131_block_008
theorem row11131_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11131 row11131_large_left_right_right :=
  Budget.RowUpperBound.branch row11131_block_007_covers row11131_block_008_covers

def row11131_large_left_right : List Nat := row11131_block_006 ++ row11131_large_left_right_right
theorem row11131_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11131 row11131_large_left_right :=
  Budget.RowUpperBound.branch row11131_block_006_covers row11131_large_left_right_right_covers

def row11131_large_left : List Nat := row11131_large_left_left ++ row11131_large_left_right
theorem row11131_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11131 row11131_large_left :=
  Budget.RowUpperBound.branch row11131_large_left_left_covers row11131_large_left_right_covers

def row11131_large_right_left : List Nat := row11131_block_009 ++ row11131_block_010
theorem row11131_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11131 row11131_large_right_left :=
  Budget.RowUpperBound.branch row11131_block_009_covers row11131_block_010_covers

def row11131_large_right_right_right : List Nat := row11131_block_012 ++ row11131_block_013
theorem row11131_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11131 row11131_large_right_right_right :=
  Budget.RowUpperBound.branch row11131_block_012_covers row11131_block_013_covers

def row11131_large_right_right : List Nat := row11131_block_011 ++ row11131_large_right_right_right
theorem row11131_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11131 row11131_large_right_right :=
  Budget.RowUpperBound.branch row11131_block_011_covers row11131_large_right_right_right_covers

def row11131_large_right : List Nat := row11131_large_right_left ++ row11131_large_right_right
theorem row11131_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11131 row11131_large_right :=
  Budget.RowUpperBound.branch row11131_large_right_left_covers row11131_large_right_right_covers

def row11131_large : List Nat := row11131_large_left ++ row11131_large_right
theorem row11131_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11131 row11131_large :=
  Budget.RowUpperBound.branch row11131_large_left_covers row11131_large_right_covers

def row11131 : List Nat := row11131_large
theorem row11131_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11131 row11131 := row11131_large_covers

def row11149_small : List Nat := []
theorem row11149_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11149 row11149_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11149_block_005 : List Nat := []
theorem row11149_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11149 row11149_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11149_block_006 : List Nat := [11383]
def plan11149_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), .first, .second, .first, .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 23), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .second, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .second, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row11149_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11149 row11149_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11149_block_006
  · decide
  · rfl

def row11149_block_007 : List Nat := [12583, 12619, 12979]
def plan11149_block_007 : List Budget.Choice := [(.firstFactor 7), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 31), .possible, (.secondFactor 23), .first, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 31), .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 13), (.firstFactor 7), (.firstFactor 17), (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, (.firstFactor 29), .first, .second, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.secondFactor 31), .first, (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 13), .first, .second, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 23), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 11), (.firstFactor 13), (.secondFactor 7), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 37), .first, .first, (.secondFactor 13), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row11149_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11149 row11149_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11149_block_007
  · decide
  · rfl

def row11149_block_008 : List Nat := [14767]
def plan11149_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, .first, .possible, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .second, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11149_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11149 row11149_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11149_block_008
  · decide
  · rfl

def row11149_block_009 : List Nat := []
theorem row11149_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11149 row11149_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11149_block_010 : List Nat := []
theorem row11149_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11149 row11149_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11149_block_011 : List Nat := []
theorem row11149_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11149 row11149_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11149_block_012 : List Nat := []
theorem row11149_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11149 row11149_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11149_block_013 : List Nat := []
theorem row11149_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11149 row11149_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11149_large_left_left : List Nat := row11149_small ++ row11149_block_005
theorem row11149_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11149 row11149_large_left_left :=
  Budget.RowUpperBound.branch row11149_small_covers row11149_block_005_covers

def row11149_large_left_right_right : List Nat := row11149_block_007 ++ row11149_block_008
theorem row11149_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11149 row11149_large_left_right_right :=
  Budget.RowUpperBound.branch row11149_block_007_covers row11149_block_008_covers

def row11149_large_left_right : List Nat := row11149_block_006 ++ row11149_large_left_right_right
theorem row11149_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11149 row11149_large_left_right :=
  Budget.RowUpperBound.branch row11149_block_006_covers row11149_large_left_right_right_covers

def row11149_large_left : List Nat := row11149_large_left_left ++ row11149_large_left_right
theorem row11149_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11149 row11149_large_left :=
  Budget.RowUpperBound.branch row11149_large_left_left_covers row11149_large_left_right_covers

def row11149_large_right_left : List Nat := row11149_block_009 ++ row11149_block_010
theorem row11149_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11149 row11149_large_right_left :=
  Budget.RowUpperBound.branch row11149_block_009_covers row11149_block_010_covers

def row11149_large_right_right_right : List Nat := row11149_block_012 ++ row11149_block_013
theorem row11149_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11149 row11149_large_right_right_right :=
  Budget.RowUpperBound.branch row11149_block_012_covers row11149_block_013_covers

def row11149_large_right_right : List Nat := row11149_block_011 ++ row11149_large_right_right_right
theorem row11149_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11149 row11149_large_right_right :=
  Budget.RowUpperBound.branch row11149_block_011_covers row11149_large_right_right_right_covers

def row11149_large_right : List Nat := row11149_large_right_left ++ row11149_large_right_right
theorem row11149_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11149 row11149_large_right :=
  Budget.RowUpperBound.branch row11149_large_right_left_covers row11149_large_right_right_covers

def row11149_large : List Nat := row11149_large_left ++ row11149_large_right
theorem row11149_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11149 row11149_large :=
  Budget.RowUpperBound.branch row11149_large_left_covers row11149_large_right_covers

def row11149 : List Nat := row11149_large
theorem row11149_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11149 row11149 := row11149_large_covers

end ProjectEulerSolutions.PrimePairLarge
