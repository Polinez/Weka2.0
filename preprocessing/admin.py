from django.contrib import admin
from .models import PreprocessingType, PreprocessingPipeline, PreprocessingStep


class PreprocessingStepInline(admin.TabularInline):
    model = PreprocessingStep
    extra = 0


@admin.register(PreprocessingType)
class PreprocessingTypeAdmin(admin.ModelAdmin):
    list_display = ('name', 'code_reference')


@admin.register(PreprocessingPipeline)
class PreprocessingPipelineAdmin(admin.ModelAdmin):
    list_display = ('id', 'dataset', 'is_active', 'created_at')
    list_filter = ('is_active',)
    inlines = [PreprocessingStepInline]
