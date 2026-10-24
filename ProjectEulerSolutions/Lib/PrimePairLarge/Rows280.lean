import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row9041_small : List Nat := []
theorem row9041_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9041 row9041_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9041_block_005 : List Nat := [9437, 9521, 9551, 9851, 10223, 10301]
def plan9041_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .possible, (.secondFactor 7), (.firstFactor 7), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), .possible, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .second, .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 17), .second, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), (.secondFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), .possible, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .second, .first, .first]
set_option maxRecDepth 2048 in
theorem row9041_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9041 row9041_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9041_block_005
  · decide
  · rfl

def row9041_block_006 : List Nat := [11027, 11069, 11093, 11987, 12203, 12239]
def plan9041_block_006 : List Budget.Choice := [.second, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 29), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), .first, .first, (.firstFactor 17), (.secondFactor 13), (.secondFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 19), (.firstFactor 23), .first, .first, (.secondFactor 7), .possible, .first, .first, .first, .possible, .first, .first, (.firstFactor 11), .possible, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 23), (.secondFactor 13), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 19), .first, (.secondFactor 29), (.firstFactor 11), .first, .first, .first, .first, .second, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 23), .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .second, .first, .second, (.firstFactor 7), (.firstFactor 29), (.firstFactor 23), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 13), (.firstFactor 19), (.firstFactor 37), .first, (.secondFactor 7), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 19), .second, (.firstFactor 11), (.firstFactor 29), .first, .first, .first, .first, .possible, .first, (.secondFactor 19), .possible, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 29), .second, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13)]
set_option maxRecDepth 2048 in
theorem row9041_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9041 row9041_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9041_block_006
  · decide
  · rfl

def row9041_block_007 : List Nat := []
def plan9041_block_007 : List Budget.Choice := [.first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 17), .first, (.secondFactor 13), (.firstFactor 23), (.firstFactor 7), .first, .first, .second, (.firstFactor 11), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 17), .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .second, .first, .first, .second, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 17), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 11), (.secondFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 13), .first, .second, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row9041_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9041 row9041_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9041_block_007
  · decide
  · rfl

def row9041_block_008 : List Nat := [15053]
def plan9041_block_008 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 7), .first, .second, (.secondFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 17), (.firstFactor 13), .first, (.secondFactor 23), (.firstFactor 19), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 29), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 13), (.firstFactor 23), .first, (.firstFactor 19), .first, .first, .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 13), .first, .possible, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 17), (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 11), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 37), (.secondFactor 11), (.secondFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 11), .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 37), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 19), (.secondFactor 13), (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 29), (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 13), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .first, (.secondFactor 11), .first, .second, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 11), (.secondFactor 19), (.firstFactor 23), .second, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9041_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9041 row9041_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9041_block_008
  · decide
  · rfl

def row9041_block_009 : List Nat := []
def plan9041_block_009 : List Budget.Choice := [.first, (.secondFactor 17), (.secondFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 29), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), (.secondFactor 19), .first, .first, .first, .second, (.secondFactor 13), .first, .first, (.secondFactor 23), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9041_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9041 row9041_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9041_block_009
  · decide
  · rfl

def row9041_block_010 : List Nat := []
theorem row9041_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9041 row9041_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9041_block_011 : List Nat := []
theorem row9041_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9041 row9041_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9041_block_012 : List Nat := []
theorem row9041_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9041 row9041_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9041_block_013 : List Nat := []
theorem row9041_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9041 row9041_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9041_large_left_left : List Nat := row9041_small ++ row9041_block_005
theorem row9041_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9041 row9041_large_left_left :=
  Budget.RowUpperBound.branch row9041_small_covers row9041_block_005_covers

def row9041_large_left_right_right : List Nat := row9041_block_007 ++ row9041_block_008
theorem row9041_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9041 row9041_large_left_right_right :=
  Budget.RowUpperBound.branch row9041_block_007_covers row9041_block_008_covers

def row9041_large_left_right : List Nat := row9041_block_006 ++ row9041_large_left_right_right
theorem row9041_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9041 row9041_large_left_right :=
  Budget.RowUpperBound.branch row9041_block_006_covers row9041_large_left_right_right_covers

def row9041_large_left : List Nat := row9041_large_left_left ++ row9041_large_left_right
theorem row9041_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9041 row9041_large_left :=
  Budget.RowUpperBound.branch row9041_large_left_left_covers row9041_large_left_right_covers

def row9041_large_right_left : List Nat := row9041_block_009 ++ row9041_block_010
theorem row9041_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9041 row9041_large_right_left :=
  Budget.RowUpperBound.branch row9041_block_009_covers row9041_block_010_covers

def row9041_large_right_right_right : List Nat := row9041_block_012 ++ row9041_block_013
theorem row9041_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9041 row9041_large_right_right_right :=
  Budget.RowUpperBound.branch row9041_block_012_covers row9041_block_013_covers

def row9041_large_right_right : List Nat := row9041_block_011 ++ row9041_large_right_right_right
theorem row9041_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9041 row9041_large_right_right :=
  Budget.RowUpperBound.branch row9041_block_011_covers row9041_large_right_right_right_covers

def row9041_large_right : List Nat := row9041_large_right_left ++ row9041_large_right_right
theorem row9041_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9041 row9041_large_right :=
  Budget.RowUpperBound.branch row9041_large_right_left_covers row9041_large_right_right_covers

def row9041_large : List Nat := row9041_large_left ++ row9041_large_right
theorem row9041_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9041 row9041_large :=
  Budget.RowUpperBound.branch row9041_large_left_covers row9041_large_right_covers

def row9041 : List Nat := row9041_large
theorem row9041_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9041 row9041 := row9041_large_covers

def row9043_small : List Nat := []
theorem row9043_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9043 row9043_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9043_block_005 : List Nat := [9157, 10093]
def plan9043_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 23), .first, (.secondFactor 7), (.secondFactor 17), .first, (.secondFactor 7), .second, (.firstFactor 17), .first, (.firstFactor 11), .possible, .first, .first, (.firstFactor 7), (.secondFactor 23), .first, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .second, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 17), (.secondFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 7), (.secondFactor 19), (.firstFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .second, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .second, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 31), (.firstFactor 13), .first, (.secondFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .possible, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7)]
set_option maxRecDepth 2048 in
theorem row9043_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9043 row9043_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9043_block_005
  · decide
  · rfl

def row9043_block_006 : List Nat := [11491, 12037]
def plan9043_block_006 : List Budget.Choice := [.first, .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.secondFactor 11), .first, .first, (.firstFactor 11), (.firstFactor 17), (.firstFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 23), (.secondFactor 11), (.firstFactor 7), .first, (.firstFactor 13), .first, (.secondFactor 13), .first, (.secondFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 19), (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 11), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), .first, .second, .first, (.secondFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 17), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 19), (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 11), .first, .first, .first, .second, (.firstFactor 19), .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9043_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9043 row9043_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9043_block_006
  · decide
  · rfl

def row9043_block_007 : List Nat := [13093, 13381, 13537, 14347]
def plan9043_block_007 : List Budget.Choice := [(.secondFactor 13), (.firstFactor 17), .first, (.firstFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 19), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 11), .first, (.secondFactor 31), (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 13), .first, .first, .second, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 37), (.secondFactor 7), (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .second, .possible, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 17), (.firstFactor 29), (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 19), (.secondFactor 7), .first, (.secondFactor 23), .first, .first, (.secondFactor 29), .first, .possible, .first, (.firstFactor 11), (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .possible, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 7), .first, (.secondFactor 17), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .second, .second, .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .possible, .first, .first, (.firstFactor 11), (.secondFactor 7), .second, .first, .first, .first, (.secondFactor 11), (.firstFactor 13), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row9043_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9043 row9043_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9043_block_007
  · decide
  · rfl

def row9043_block_008 : List Nat := [14533, 15991]
def plan9043_block_008 : List Budget.Choice := [.first, .first, .second, .first, .possible, .first, .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.firstFactor 23), .first, .first, (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .second, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 19), (.secondFactor 23), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 31), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 29), (.firstFactor 29), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .possible, .first, .first, (.secondFactor 19), .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, .first, .first, .second, .second, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), (.firstFactor 11), (.firstFactor 19), (.firstFactor 17), (.secondFactor 7), .first, .first, .first, .second, .first, .second, (.secondFactor 11), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9043_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9043 row9043_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9043_block_008
  · decide
  · rfl

def row9043_block_009 : List Nat := [16561]
def plan9043_block_009 : List Budget.Choice := [(.firstFactor 37), .first, .first, .first, .possible, (.firstFactor 7), .second, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9043_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9043 row9043_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9043_block_009
  · decide
  · rfl

def row9043_block_010 : List Nat := []
theorem row9043_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9043 row9043_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9043_block_011 : List Nat := []
theorem row9043_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9043 row9043_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9043_block_012 : List Nat := []
theorem row9043_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9043 row9043_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9043_block_013 : List Nat := []
theorem row9043_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9043 row9043_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9043_large_left_left : List Nat := row9043_small ++ row9043_block_005
theorem row9043_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9043 row9043_large_left_left :=
  Budget.RowUpperBound.branch row9043_small_covers row9043_block_005_covers

def row9043_large_left_right_right : List Nat := row9043_block_007 ++ row9043_block_008
theorem row9043_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9043 row9043_large_left_right_right :=
  Budget.RowUpperBound.branch row9043_block_007_covers row9043_block_008_covers

def row9043_large_left_right : List Nat := row9043_block_006 ++ row9043_large_left_right_right
theorem row9043_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9043 row9043_large_left_right :=
  Budget.RowUpperBound.branch row9043_block_006_covers row9043_large_left_right_right_covers

def row9043_large_left : List Nat := row9043_large_left_left ++ row9043_large_left_right
theorem row9043_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9043 row9043_large_left :=
  Budget.RowUpperBound.branch row9043_large_left_left_covers row9043_large_left_right_covers

def row9043_large_right_left : List Nat := row9043_block_009 ++ row9043_block_010
theorem row9043_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9043 row9043_large_right_left :=
  Budget.RowUpperBound.branch row9043_block_009_covers row9043_block_010_covers

def row9043_large_right_right_right : List Nat := row9043_block_012 ++ row9043_block_013
theorem row9043_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9043 row9043_large_right_right_right :=
  Budget.RowUpperBound.branch row9043_block_012_covers row9043_block_013_covers

def row9043_large_right_right : List Nat := row9043_block_011 ++ row9043_large_right_right_right
theorem row9043_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9043 row9043_large_right_right :=
  Budget.RowUpperBound.branch row9043_block_011_covers row9043_large_right_right_right_covers

def row9043_large_right : List Nat := row9043_large_right_left ++ row9043_large_right_right
theorem row9043_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9043 row9043_large_right :=
  Budget.RowUpperBound.branch row9043_large_right_left_covers row9043_large_right_right_covers

def row9043_large : List Nat := row9043_large_left ++ row9043_large_right
theorem row9043_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9043 row9043_large :=
  Budget.RowUpperBound.branch row9043_large_left_covers row9043_large_right_covers

def row9043 : List Nat := row9043_large
theorem row9043_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9043 row9043 := row9043_large_covers

def row9049_small : List Nat := []
theorem row9049_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9049 row9049_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9049_block_005 : List Nat := [9187, 9613, 9973]
def plan9049_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 13), .first, (.firstFactor 17), .first, (.secondFactor 23), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, (.firstFactor 23), .possible, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 23), .first, (.firstFactor 19), .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.secondFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 19), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .possible, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 13), (.firstFactor 23), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), (.secondFactor 19), (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 37), (.firstFactor 7), .first, .first, .first, .first, .second, (.firstFactor 7), .second, .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, (.secondFactor 13), .first, .possible, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.secondFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9049_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9049 row9049_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9049_block_005
  · decide
  · rfl

def row9049_block_006 : List Nat := [10627, 11047, 11083, 11311, 11383, 11551, 11863, 12097]
def plan9049_block_006 : List Budget.Choice := [.first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 23), .first, .first, .first, .possible, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 29), .first, .second, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 17), .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, (.firstFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), .second, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, .possible, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .second, .first, (.firstFactor 13), .first, .possible, .first, (.secondFactor 11), .second, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 11), (.firstFactor 31), (.secondFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .possible, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 13), .possible, .first, .first, .first, .first, .first, .first, .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 31), .first, .first, .first, .first, .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9049_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9049 row9049_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9049_block_006
  · decide
  · rfl

def row9049_block_007 : List Nat := [12451, 12637]
def plan9049_block_007 : List Budget.Choice := [(.firstFactor 7), (.firstFactor 29), .first, .possible, (.secondFactor 7), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 11), (.secondFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 11), (.secondFactor 7), (.firstFactor 7), (.firstFactor 17), .first, (.firstFactor 11), (.firstFactor 13), .possible, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 17), .first, (.secondFactor 23), .second, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 17), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 13), (.secondFactor 23), (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.secondFactor 11), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 23), .first, (.secondFactor 13), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.secondFactor 29), .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 29), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 11), .first, .first, (.firstFactor 23), .first, (.secondFactor 19), (.firstFactor 29), .first, .first, (.secondFactor 7), .second, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 11)]
set_option maxRecDepth 2048 in
theorem row9049_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9049 row9049_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9049_block_007
  · decide
  · rfl

def row9049_block_008 : List Nat := [14533, 14713, 15241, 16231, 16417]
def plan9049_block_008 : List Budget.Choice := [(.firstFactor 7), .first, (.secondFactor 31), .first, .possible, .first, .first, .first, (.secondFactor 13), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .possible, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 19), .first, (.secondFactor 11), (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 29), .first, (.secondFactor 11), .first, .first, (.firstFactor 11), (.secondFactor 17), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 23), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .possible, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, (.firstFactor 23), .first, .second, (.secondFactor 7), (.firstFactor 7), .first, (.secondFactor 13), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 17), .first, .first, (.firstFactor 11), .first, (.secondFactor 7), (.secondFactor 23), .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.firstFactor 19), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 7), (.firstFactor 11), (.firstFactor 37), .first, (.secondFactor 19), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 37), .first, .first, (.secondFactor 13), .first, .first, .first, .first, .possible, (.firstFactor 19), .first, (.secondFactor 13), (.firstFactor 17), .first, .first, (.secondFactor 29), (.secondFactor 11), .first, .first, (.firstFactor 13), (.firstFactor 7), (.firstFactor 31), (.firstFactor 7), .possible, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row9049_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9049 row9049_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9049_block_008
  · decide
  · rfl

def row9049_block_009 : List Nat := [16633]
def plan9049_block_009 : List Budget.Choice := [(.firstFactor 13), .first, .first, .first, (.secondFactor 17), (.firstFactor 31), (.firstFactor 11), (.secondFactor 11), .first, .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 19), .first, .second, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9049_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9049 row9049_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9049_block_009
  · decide
  · rfl

def row9049_block_010 : List Nat := []
theorem row9049_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9049 row9049_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9049_block_011 : List Nat := []
theorem row9049_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9049 row9049_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9049_block_012 : List Nat := []
theorem row9049_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9049 row9049_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9049_block_013 : List Nat := []
theorem row9049_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9049 row9049_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9049_large_left_left : List Nat := row9049_small ++ row9049_block_005
theorem row9049_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9049 row9049_large_left_left :=
  Budget.RowUpperBound.branch row9049_small_covers row9049_block_005_covers

def row9049_large_left_right_right : List Nat := row9049_block_007 ++ row9049_block_008
theorem row9049_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9049 row9049_large_left_right_right :=
  Budget.RowUpperBound.branch row9049_block_007_covers row9049_block_008_covers

def row9049_large_left_right : List Nat := row9049_block_006 ++ row9049_large_left_right_right
theorem row9049_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9049 row9049_large_left_right :=
  Budget.RowUpperBound.branch row9049_block_006_covers row9049_large_left_right_right_covers

def row9049_large_left : List Nat := row9049_large_left_left ++ row9049_large_left_right
theorem row9049_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9049 row9049_large_left :=
  Budget.RowUpperBound.branch row9049_large_left_left_covers row9049_large_left_right_covers

def row9049_large_right_left : List Nat := row9049_block_009 ++ row9049_block_010
theorem row9049_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9049 row9049_large_right_left :=
  Budget.RowUpperBound.branch row9049_block_009_covers row9049_block_010_covers

def row9049_large_right_right_right : List Nat := row9049_block_012 ++ row9049_block_013
theorem row9049_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9049 row9049_large_right_right_right :=
  Budget.RowUpperBound.branch row9049_block_012_covers row9049_block_013_covers

def row9049_large_right_right : List Nat := row9049_block_011 ++ row9049_large_right_right_right
theorem row9049_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9049 row9049_large_right_right :=
  Budget.RowUpperBound.branch row9049_block_011_covers row9049_large_right_right_right_covers

def row9049_large_right : List Nat := row9049_large_right_left ++ row9049_large_right_right
theorem row9049_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9049 row9049_large_right :=
  Budget.RowUpperBound.branch row9049_large_right_left_covers row9049_large_right_right_covers

def row9049_large : List Nat := row9049_large_left ++ row9049_large_right
theorem row9049_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9049 row9049_large :=
  Budget.RowUpperBound.branch row9049_large_left_covers row9049_large_right_covers

def row9049 : List Nat := row9049_large
theorem row9049_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9049 row9049 := row9049_large_covers

def row9059_small : List Nat := []
theorem row9059_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 9059 row9059_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row9059_block_005 : List Nat := [9473, 10151]
def plan9059_block_005 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 13), (.secondFactor 37), (.secondFactor 23), (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .second, .first, (.firstFactor 23), .first, (.firstFactor 7), .possible, .first, (.secondFactor 29), (.secondFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .second, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 17), (.secondFactor 17), .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 17), (.firstFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, .possible, .first, .first, (.secondFactor 7), .first, (.firstFactor 11), (.secondFactor 23), (.secondFactor 7), (.firstFactor 23), .first, (.firstFactor 11), (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 13), .first, .first]
set_option maxRecDepth 2048 in
theorem row9059_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 9059 row9059_block_005 := by
  apply Budget.row_upper_of_plan _ _ _ plan9059_block_005
  · decide
  · rfl

def row9059_block_006 : List Nat := [10391, 10781, 11411, 11777]
def plan9059_block_006 : List Budget.Choice := [.possible, .first, (.firstFactor 31), .first, (.firstFactor 17), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, .first, (.firstFactor 7), (.firstFactor 17), .first, .first, .possible, .first, (.firstFactor 31), .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, .first, (.firstFactor 11), .second, .first, .first, (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, (.secondFactor 11), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 7), .second, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 19), (.secondFactor 17), .possible, (.secondFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .second, .first, (.secondFactor 13), .first, (.secondFactor 29), .first, .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .possible, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .second, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 29), (.secondFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 23), .first, (.secondFactor 17), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 37), .first, (.secondFactor 17), (.firstFactor 7), (.secondFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 23), (.firstFactor 17), .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row9059_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 9059 row9059_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan9059_block_006
  · decide
  · rfl

def row9059_block_007 : List Nat := [12653, 13691, 13751, 13913]
def plan9059_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), (.secondFactor 7), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.secondFactor 13), (.firstFactor 19), .possible, (.firstFactor 31), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, .first, .first, .first, .first, .second, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 11), .first, .first, .second, (.firstFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 19), (.secondFactor 7), (.firstFactor 23), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 23), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.secondFactor 7), (.secondFactor 13), .first, .second, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .possible, .first, (.firstFactor 13), .first, .first, .first, .first, .first, .possible, (.firstFactor 7), .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 17), .first, (.secondFactor 7), .possible, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 11), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), (.secondFactor 23), .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 17), (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 37), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 19), .first, .first]
set_option maxRecDepth 2048 in
theorem row9059_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 9059 row9059_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan9059_block_007
  · decide
  · rfl

def row9059_block_008 : List Nat := [14627, 15401, 15749, 16139]
def plan9059_block_008 : List Budget.Choice := [.first, .first, .first, (.firstFactor 31), .first, (.firstFactor 11), .second, (.firstFactor 37), .first, .first, (.secondFactor 17), .first, (.secondFactor 11), .first, (.secondFactor 7), .possible, .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 11), (.firstFactor 11), .first, (.firstFactor 19), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, .second, (.secondFactor 7), .first, .first, .first, (.firstFactor 37), .first, .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .second, .first, .first, .second, .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.firstFactor 31), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 29), .first, .first, (.secondFactor 23), (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 19), .first, .possible, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 23), (.secondFactor 19), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 13), (.secondFactor 11), .first, .first, .first, .first, (.secondFactor 7), .second, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .possible, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), .second, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .second, .first, .first, .first, (.firstFactor 29), .first, .first, .second, .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 23), .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 13), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row9059_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 9059 row9059_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan9059_block_008
  · decide
  · rfl

def row9059_block_009 : List Nat := [16631]
def plan9059_block_009 : List Budget.Choice := [.first, (.firstFactor 7), (.firstFactor 37), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 31), (.secondFactor 13), .possible, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .second, .first, .first, (.firstFactor 29), .first, .first, .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.secondFactor 7), (.secondFactor 11), .first, .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row9059_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 9059 row9059_block_009 := by
  apply Budget.row_upper_of_plan _ _ _ plan9059_block_009
  · decide
  · rfl

def row9059_block_010 : List Nat := []
theorem row9059_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 9059 row9059_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row9059_block_011 : List Nat := []
theorem row9059_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 9059 row9059_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row9059_block_012 : List Nat := []
theorem row9059_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 9059 row9059_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row9059_block_013 : List Nat := []
theorem row9059_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 9059 row9059_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row9059_large_left_left : List Nat := row9059_small ++ row9059_block_005
theorem row9059_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 9059 row9059_large_left_left :=
  Budget.RowUpperBound.branch row9059_small_covers row9059_block_005_covers

def row9059_large_left_right_right : List Nat := row9059_block_007 ++ row9059_block_008
theorem row9059_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 9059 row9059_large_left_right_right :=
  Budget.RowUpperBound.branch row9059_block_007_covers row9059_block_008_covers

def row9059_large_left_right : List Nat := row9059_block_006 ++ row9059_large_left_right_right
theorem row9059_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 9059 row9059_large_left_right :=
  Budget.RowUpperBound.branch row9059_block_006_covers row9059_large_left_right_right_covers

def row9059_large_left : List Nat := row9059_large_left_left ++ row9059_large_left_right
theorem row9059_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 9059 row9059_large_left :=
  Budget.RowUpperBound.branch row9059_large_left_left_covers row9059_large_left_right_covers

def row9059_large_right_left : List Nat := row9059_block_009 ++ row9059_block_010
theorem row9059_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 9059 row9059_large_right_left :=
  Budget.RowUpperBound.branch row9059_block_009_covers row9059_block_010_covers

def row9059_large_right_right_right : List Nat := row9059_block_012 ++ row9059_block_013
theorem row9059_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 9059 row9059_large_right_right_right :=
  Budget.RowUpperBound.branch row9059_block_012_covers row9059_block_013_covers

def row9059_large_right_right : List Nat := row9059_block_011 ++ row9059_large_right_right_right
theorem row9059_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 9059 row9059_large_right_right :=
  Budget.RowUpperBound.branch row9059_block_011_covers row9059_large_right_right_right_covers

def row9059_large_right : List Nat := row9059_large_right_left ++ row9059_large_right_right
theorem row9059_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 9059 row9059_large_right :=
  Budget.RowUpperBound.branch row9059_large_right_left_covers row9059_large_right_right_covers

def row9059_large : List Nat := row9059_large_left ++ row9059_large_right
theorem row9059_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 9059 row9059_large :=
  Budget.RowUpperBound.branch row9059_large_left_covers row9059_large_right_covers

def row9059 : List Nat := row9059_large
theorem row9059_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 9059 row9059 := row9059_large_covers

end ProjectEulerSolutions.PrimePairLarge
