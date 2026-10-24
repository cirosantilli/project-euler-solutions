import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11273_small : List Nat := []
theorem row11273_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11273 row11273_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11273_block_005 : List Nat := []
theorem row11273_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11273 row11273_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11273_block_006 : List Nat := [11579, 11777, 12113]
def plan11273_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 29), .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 23), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 37), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, (.firstFactor 13), .second, (.firstFactor 7), (.firstFactor 31), .second, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 31), (.firstFactor 23), (.secondFactor 23), .first, (.firstFactor 7), (.firstFactor 11), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.secondFactor 37), .first, .first, (.secondFactor 7), .first, .first, .possible, .second, (.firstFactor 17), (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 29)]
set_option maxRecDepth 2048 in
theorem row11273_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11273 row11273_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11273_block_006
  · decide
  · rfl

def row11273_block_007 : List Nat := [13337, 13463, 13829]
def plan11273_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .second, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), (.firstFactor 11), (.secondFactor 19), (.secondFactor 7), .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 13), (.firstFactor 11), (.secondFactor 19), .first, (.firstFactor 37), .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 29), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), .possible, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.secondFactor 29), .possible, (.firstFactor 17), .first, (.secondFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 23), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.secondFactor 13), .first, .possible, .first, (.secondFactor 23), (.firstFactor 7), .first, (.firstFactor 17), .first, (.secondFactor 17), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .second, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 11), (.firstFactor 23), .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 31), .first, (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row11273_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11273 row11273_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11273_block_007
  · decide
  · rfl

def row11273_block_008 : List Nat := [14633, 14759]
def plan11273_block_008 : List Budget.Choice := [.first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, .possible, (.firstFactor 11), .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .possible, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11273_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11273 row11273_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11273_block_008
  · decide
  · rfl

def row11273_block_009 : List Nat := []
theorem row11273_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11273 row11273_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11273_block_010 : List Nat := []
theorem row11273_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11273 row11273_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11273_block_011 : List Nat := []
theorem row11273_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11273 row11273_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11273_block_012 : List Nat := []
theorem row11273_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11273 row11273_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11273_block_013 : List Nat := []
theorem row11273_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11273 row11273_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11273_large_left_left : List Nat := row11273_small ++ row11273_block_005
theorem row11273_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11273 row11273_large_left_left :=
  Budget.RowUpperBound.branch row11273_small_covers row11273_block_005_covers

def row11273_large_left_right_right : List Nat := row11273_block_007 ++ row11273_block_008
theorem row11273_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11273 row11273_large_left_right_right :=
  Budget.RowUpperBound.branch row11273_block_007_covers row11273_block_008_covers

def row11273_large_left_right : List Nat := row11273_block_006 ++ row11273_large_left_right_right
theorem row11273_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11273 row11273_large_left_right :=
  Budget.RowUpperBound.branch row11273_block_006_covers row11273_large_left_right_right_covers

def row11273_large_left : List Nat := row11273_large_left_left ++ row11273_large_left_right
theorem row11273_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11273 row11273_large_left :=
  Budget.RowUpperBound.branch row11273_large_left_left_covers row11273_large_left_right_covers

def row11273_large_right_left : List Nat := row11273_block_009 ++ row11273_block_010
theorem row11273_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11273 row11273_large_right_left :=
  Budget.RowUpperBound.branch row11273_block_009_covers row11273_block_010_covers

def row11273_large_right_right_right : List Nat := row11273_block_012 ++ row11273_block_013
theorem row11273_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11273 row11273_large_right_right_right :=
  Budget.RowUpperBound.branch row11273_block_012_covers row11273_block_013_covers

def row11273_large_right_right : List Nat := row11273_block_011 ++ row11273_large_right_right_right
theorem row11273_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11273 row11273_large_right_right :=
  Budget.RowUpperBound.branch row11273_block_011_covers row11273_large_right_right_right_covers

def row11273_large_right : List Nat := row11273_large_right_left ++ row11273_large_right_right
theorem row11273_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11273 row11273_large_right :=
  Budget.RowUpperBound.branch row11273_large_right_left_covers row11273_large_right_right_covers

def row11273_large : List Nat := row11273_large_left ++ row11273_large_right
theorem row11273_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11273 row11273_large :=
  Budget.RowUpperBound.branch row11273_large_left_covers row11273_large_right_covers

def row11273 : List Nat := row11273_large
theorem row11273_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11273 row11273 := row11273_large_covers

def row11279_small : List Nat := []
theorem row11279_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11279 row11279_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11279_block_005 : List Nat := []
theorem row11279_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11279 row11279_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11279_block_006 : List Nat := [11549]
def plan11279_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .possible, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.secondFactor 7), (.secondFactor 13), (.firstFactor 31), .first, (.firstFactor 13), .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, .first, .second, (.firstFactor 23), (.firstFactor 17), (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 7), (.firstFactor 17), (.secondFactor 19), .first, (.firstFactor 23), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .second, .first, (.secondFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 13), (.firstFactor 7), .first, .first, .second, (.firstFactor 19), (.firstFactor 7), .first, .second, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 31), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), .first, (.secondFactor 29), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 17)]
set_option maxRecDepth 2048 in
theorem row11279_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11279 row11279_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11279_block_006
  · decide
  · rfl

def row11279_block_007 : List Nat := [13613, 13691, 13781]
def plan11279_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 17), .first, .first, .first, (.firstFactor 23), .first, (.secondFactor 13), (.secondFactor 23), (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 17), (.secondFactor 13), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 29), .first, .first, (.firstFactor 19), .first, .first, (.secondFactor 17), .first, .first, .first, .first, .first, .second, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), (.secondFactor 37), .first, (.secondFactor 23), .first, .first, (.firstFactor 17), (.firstFactor 19), .second, (.firstFactor 31), .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 29), (.secondFactor 29), (.secondFactor 7), .first, .first, .first, .second, .first, (.secondFactor 37), (.firstFactor 11), .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, (.secondFactor 23), (.firstFactor 37), .first, (.firstFactor 17), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), (.secondFactor 23), (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .possible, .first, .second, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 17), (.firstFactor 23), .first, (.secondFactor 23), .first, (.firstFactor 37), .second, .first, (.firstFactor 29), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 19), (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, (.secondFactor 23), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 19), .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first]
set_option maxRecDepth 2048 in
theorem row11279_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11279 row11279_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11279_block_007
  · decide
  · rfl

def row11279_block_008 : List Nat := [14747]
def plan11279_block_008 : List Budget.Choice := [.first, (.firstFactor 13), .first, (.secondFactor 7), .first, (.firstFactor 19), (.firstFactor 7), (.firstFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .second, .first, .second, (.firstFactor 13), .first, .first, .first, .possible, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11279_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11279 row11279_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11279_block_008
  · decide
  · rfl

def row11279_block_009 : List Nat := []
theorem row11279_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11279 row11279_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11279_block_010 : List Nat := []
theorem row11279_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11279 row11279_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11279_block_011 : List Nat := []
theorem row11279_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11279 row11279_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11279_block_012 : List Nat := []
theorem row11279_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11279 row11279_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11279_block_013 : List Nat := []
theorem row11279_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11279 row11279_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11279_large_left_left : List Nat := row11279_small ++ row11279_block_005
theorem row11279_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11279 row11279_large_left_left :=
  Budget.RowUpperBound.branch row11279_small_covers row11279_block_005_covers

def row11279_large_left_right_right : List Nat := row11279_block_007 ++ row11279_block_008
theorem row11279_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11279 row11279_large_left_right_right :=
  Budget.RowUpperBound.branch row11279_block_007_covers row11279_block_008_covers

def row11279_large_left_right : List Nat := row11279_block_006 ++ row11279_large_left_right_right
theorem row11279_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11279 row11279_large_left_right :=
  Budget.RowUpperBound.branch row11279_block_006_covers row11279_large_left_right_right_covers

def row11279_large_left : List Nat := row11279_large_left_left ++ row11279_large_left_right
theorem row11279_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11279 row11279_large_left :=
  Budget.RowUpperBound.branch row11279_large_left_left_covers row11279_large_left_right_covers

def row11279_large_right_left : List Nat := row11279_block_009 ++ row11279_block_010
theorem row11279_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11279 row11279_large_right_left :=
  Budget.RowUpperBound.branch row11279_block_009_covers row11279_block_010_covers

def row11279_large_right_right_right : List Nat := row11279_block_012 ++ row11279_block_013
theorem row11279_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11279 row11279_large_right_right_right :=
  Budget.RowUpperBound.branch row11279_block_012_covers row11279_block_013_covers

def row11279_large_right_right : List Nat := row11279_block_011 ++ row11279_large_right_right_right
theorem row11279_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11279 row11279_large_right_right :=
  Budget.RowUpperBound.branch row11279_block_011_covers row11279_large_right_right_right_covers

def row11279_large_right : List Nat := row11279_large_right_left ++ row11279_large_right_right
theorem row11279_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11279 row11279_large_right :=
  Budget.RowUpperBound.branch row11279_large_right_left_covers row11279_large_right_right_covers

def row11279_large : List Nat := row11279_large_left ++ row11279_large_right
theorem row11279_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11279 row11279_large :=
  Budget.RowUpperBound.branch row11279_large_left_covers row11279_large_right_covers

def row11279 : List Nat := row11279_large
theorem row11279_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11279 row11279 := row11279_large_covers

def row11287_small : List Nat := []
theorem row11287_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11287 row11287_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11287_block_005 : List Nat := []
theorem row11287_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11287 row11287_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11287_block_006 : List Nat := []
def plan11287_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, .first, .second, .first, (.secondFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 17), (.firstFactor 7), .second, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), (.secondFactor 31), .first, .first, (.secondFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), (.firstFactor 31), (.firstFactor 23), .first, (.secondFactor 17), .first, .first, .first, .first, .second, .first, (.firstFactor 19), .first, .first, (.secondFactor 7), (.firstFactor 23), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 19), .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 37), .first, .first, (.firstFactor 11), .first, (.firstFactor 23), .first, (.secondFactor 17), .second, .first, (.firstFactor 11), .first]
set_option maxRecDepth 2048 in
theorem row11287_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11287 row11287_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11287_block_006
  · decide
  · rfl

def row11287_block_007 : List Nat := [12457, 12919, 14281]
def plan11287_block_007 : List Budget.Choice := [.first, (.firstFactor 17), .first, (.firstFactor 29), .possible, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 11), .first, .second, .first, (.secondFactor 7), (.secondFactor 17), .second, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 31), .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 17), (.secondFactor 13), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .possible, .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 29), .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 23), (.secondFactor 17), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), (.secondFactor 7), (.firstFactor 19), .second, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 7), .first, .second, .first, (.secondFactor 19), (.firstFactor 11), .first, .first, (.firstFactor 31), (.secondFactor 13), .first, (.firstFactor 29), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.secondFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.secondFactor 17), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, .first, (.secondFactor 7), .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 37), (.secondFactor 29), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 13), (.firstFactor 19), .first, .first, (.firstFactor 7)]
set_option maxRecDepth 2048 in
theorem row11287_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11287 row11287_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11287_block_007
  · decide
  · rfl

def row11287_block_008 : List Nat := []
def plan11287_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 31), .first, (.firstFactor 37), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .second, .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11287_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11287 row11287_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11287_block_008
  · decide
  · rfl

def row11287_block_009 : List Nat := []
theorem row11287_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11287 row11287_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11287_block_010 : List Nat := []
theorem row11287_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11287 row11287_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11287_block_011 : List Nat := []
theorem row11287_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11287 row11287_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11287_block_012 : List Nat := []
theorem row11287_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11287 row11287_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11287_block_013 : List Nat := []
theorem row11287_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11287 row11287_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11287_large_left_left : List Nat := row11287_small ++ row11287_block_005
theorem row11287_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11287 row11287_large_left_left :=
  Budget.RowUpperBound.branch row11287_small_covers row11287_block_005_covers

def row11287_large_left_right_right : List Nat := row11287_block_007 ++ row11287_block_008
theorem row11287_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11287 row11287_large_left_right_right :=
  Budget.RowUpperBound.branch row11287_block_007_covers row11287_block_008_covers

def row11287_large_left_right : List Nat := row11287_block_006 ++ row11287_large_left_right_right
theorem row11287_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11287 row11287_large_left_right :=
  Budget.RowUpperBound.branch row11287_block_006_covers row11287_large_left_right_right_covers

def row11287_large_left : List Nat := row11287_large_left_left ++ row11287_large_left_right
theorem row11287_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11287 row11287_large_left :=
  Budget.RowUpperBound.branch row11287_large_left_left_covers row11287_large_left_right_covers

def row11287_large_right_left : List Nat := row11287_block_009 ++ row11287_block_010
theorem row11287_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11287 row11287_large_right_left :=
  Budget.RowUpperBound.branch row11287_block_009_covers row11287_block_010_covers

def row11287_large_right_right_right : List Nat := row11287_block_012 ++ row11287_block_013
theorem row11287_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11287 row11287_large_right_right_right :=
  Budget.RowUpperBound.branch row11287_block_012_covers row11287_block_013_covers

def row11287_large_right_right : List Nat := row11287_block_011 ++ row11287_large_right_right_right
theorem row11287_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11287 row11287_large_right_right :=
  Budget.RowUpperBound.branch row11287_block_011_covers row11287_large_right_right_right_covers

def row11287_large_right : List Nat := row11287_large_right_left ++ row11287_large_right_right
theorem row11287_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11287 row11287_large_right :=
  Budget.RowUpperBound.branch row11287_large_right_left_covers row11287_large_right_right_covers

def row11287_large : List Nat := row11287_large_left ++ row11287_large_right
theorem row11287_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11287 row11287_large :=
  Budget.RowUpperBound.branch row11287_large_left_covers row11287_large_right_covers

def row11287 : List Nat := row11287_large
theorem row11287_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11287 row11287 := row11287_large_covers

def row11299_small : List Nat := []
theorem row11299_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11299 row11299_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11299_block_005 : List Nat := []
theorem row11299_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11299 row11299_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11299_block_006 : List Nat := []
def plan11299_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.firstFactor 29), (.firstFactor 23), .first, (.secondFactor 13), .first, .second, .first, (.secondFactor 19), .first, .first, .first, .first, (.firstFactor 19), (.firstFactor 17), .first, (.firstFactor 31), .first, .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 7), (.firstFactor 23), .first, (.secondFactor 7), .first, .first, .first, .second, .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 13), (.secondFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .first, .second, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 31), .first, .first, (.secondFactor 17), .first, (.firstFactor 37), .first, .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 19), .first]
set_option maxRecDepth 2048 in
theorem row11299_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11299 row11299_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11299_block_006
  · decide
  · rfl

def row11299_block_007 : List Nat := [12757, 12799, 13009]
def plan11299_block_007 : List Budget.Choice := [(.firstFactor 11), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.secondFactor 17), (.firstFactor 7), .first, (.firstFactor 19), (.firstFactor 31), (.firstFactor 13), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .possible, (.firstFactor 7), .first, .first, .possible, .first, .first, (.firstFactor 13), .first, .second, (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 7), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .possible, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 17), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 7), (.firstFactor 13), (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, (.firstFactor 17), .first, (.firstFactor 29), (.firstFactor 11), .first, .first, (.secondFactor 31), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 31), .first, (.secondFactor 23), .first, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 31), (.secondFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 13), (.firstFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 29), .first, .first, .second, .first, .first, (.secondFactor 31), .first, (.secondFactor 7), (.firstFactor 37), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, .first, (.firstFactor 19), .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.firstFactor 37), (.secondFactor 17), (.secondFactor 13), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 7), .second, .first, (.secondFactor 13), (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row11299_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11299 row11299_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11299_block_007
  · decide
  · rfl

def row11299_block_008 : List Nat := [14551]
def plan11299_block_008 : List Budget.Choice := [.first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .possible, (.secondFactor 7), .first, (.secondFactor 29), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 11), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11299_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11299 row11299_block_008 := by
  apply Budget.row_upper_of_plan _ _ _ plan11299_block_008
  · decide
  · rfl

def row11299_block_009 : List Nat := []
theorem row11299_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11299 row11299_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11299_block_010 : List Nat := []
theorem row11299_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11299 row11299_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11299_block_011 : List Nat := []
theorem row11299_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11299 row11299_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11299_block_012 : List Nat := []
theorem row11299_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11299 row11299_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11299_block_013 : List Nat := []
theorem row11299_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11299 row11299_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11299_large_left_left : List Nat := row11299_small ++ row11299_block_005
theorem row11299_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11299 row11299_large_left_left :=
  Budget.RowUpperBound.branch row11299_small_covers row11299_block_005_covers

def row11299_large_left_right_right : List Nat := row11299_block_007 ++ row11299_block_008
theorem row11299_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11299 row11299_large_left_right_right :=
  Budget.RowUpperBound.branch row11299_block_007_covers row11299_block_008_covers

def row11299_large_left_right : List Nat := row11299_block_006 ++ row11299_large_left_right_right
theorem row11299_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11299 row11299_large_left_right :=
  Budget.RowUpperBound.branch row11299_block_006_covers row11299_large_left_right_right_covers

def row11299_large_left : List Nat := row11299_large_left_left ++ row11299_large_left_right
theorem row11299_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11299 row11299_large_left :=
  Budget.RowUpperBound.branch row11299_large_left_left_covers row11299_large_left_right_covers

def row11299_large_right_left : List Nat := row11299_block_009 ++ row11299_block_010
theorem row11299_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11299 row11299_large_right_left :=
  Budget.RowUpperBound.branch row11299_block_009_covers row11299_block_010_covers

def row11299_large_right_right_right : List Nat := row11299_block_012 ++ row11299_block_013
theorem row11299_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11299 row11299_large_right_right_right :=
  Budget.RowUpperBound.branch row11299_block_012_covers row11299_block_013_covers

def row11299_large_right_right : List Nat := row11299_block_011 ++ row11299_large_right_right_right
theorem row11299_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11299 row11299_large_right_right :=
  Budget.RowUpperBound.branch row11299_block_011_covers row11299_large_right_right_right_covers

def row11299_large_right : List Nat := row11299_large_right_left ++ row11299_large_right_right
theorem row11299_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11299 row11299_large_right :=
  Budget.RowUpperBound.branch row11299_large_right_left_covers row11299_large_right_right_covers

def row11299_large : List Nat := row11299_large_left ++ row11299_large_right
theorem row11299_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11299 row11299_large :=
  Budget.RowUpperBound.branch row11299_large_left_covers row11299_large_right_covers

def row11299 : List Nat := row11299_large
theorem row11299_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11299 row11299 := row11299_large_covers

end ProjectEulerSolutions.PrimePairLarge
