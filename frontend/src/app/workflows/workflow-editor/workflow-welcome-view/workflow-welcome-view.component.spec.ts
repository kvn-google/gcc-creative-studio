/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {ComponentFixture, TestBed} from '@angular/core/testing';
import {MatDialog} from '@angular/material/dialog';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {of} from 'rxjs';
import {
  NodeTypes,
  StepStatusEnum,
  WorkflowTemplate,
} from '../../workflow.models';
import {WorkflowService} from '../../workflow.service';
import {
  WelcomeTab,
  WorkflowWelcomeViewComponent,
} from './workflow-welcome-view.component';

describe('WorkflowWelcomeViewComponent', () => {
  let component: WorkflowWelcomeViewComponent;
  let fixture: ComponentFixture<WorkflowWelcomeViewComponent>;
  let mockWorkflowService: jasmine.SpyObj<WorkflowService>;
  let mockDialog: jasmine.SpyObj<MatDialog>;

  const samplePredefinedTemplate: WorkflowTemplate = {
    id: 'predefined-1',
    name: 'Model Outfit Color Editor',
    description: 'Edits suit color',
    isPredefined: true,
    steps: [
      {
        stepId: 'user_input',
        type: NodeTypes.USER_INPUT,
        status: StepStatusEnum.IDLE,
        inputs: {},
        outputs: {},
        settings: {},
      },
    ],
  };

  const sampleUserTemplate: WorkflowTemplate = {
    id: 'user-tmpl-1',
    name: 'My Custom Template',
    description: 'Custom description',
    isPredefined: false,
    steps: [],
    createdAt: '2026-09-08T12:00:00Z',
  };

  beforeEach(async () => {
    mockWorkflowService = jasmine.createSpyObj('WorkflowService', [
      'getPredefinedTemplates',
      'getUserTemplates',
      'deleteTemplate',
    ]);
    mockDialog = jasmine.createSpyObj('MatDialog', ['open']);

    mockWorkflowService.getPredefinedTemplates.and.returnValue([
      samplePredefinedTemplate,
    ]);
    mockWorkflowService.getUserTemplates.and.returnValue(
      of([sampleUserTemplate]),
    );

    await TestBed.configureTestingModule({
      declarations: [WorkflowWelcomeViewComponent],
      imports: [MatIconModule, MatProgressSpinnerModule],
      providers: [
        {provide: WorkflowService, useValue: mockWorkflowService},
        {provide: MatDialog, useValue: mockDialog},
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(WorkflowWelcomeViewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create and load templates', () => {
    expect(component).toBeTruthy();
    expect(component.predefinedTemplates().length).toBe(1);
    expect(component.userTemplates().length).toBe(1);
    expect(component.totalTemplateCount()).toBe(2);
  });

  it('should emit null when blank workflow is selected', () => {
    spyOn(component.templateSelected, 'emit');
    component.selectBlankWorkflow();
    expect(component.templateSelected.emit).toHaveBeenCalledWith(null);
  });

  it('should emit template when a template is selected', () => {
    spyOn(component.templateSelected, 'emit');
    component.selectTemplate(samplePredefinedTemplate);
    expect(component.templateSelected.emit).toHaveBeenCalledWith(
      samplePredefinedTemplate,
    );
  });

  it('should filter templates based on search query', () => {
    component.onSearchChange('outfit');
    expect(component.filteredPredefinedTemplates().length).toBe(1);
    expect(component.filteredUserTemplates().length).toBe(0);

    component.onSearchChange('custom');
    expect(component.filteredPredefinedTemplates().length).toBe(0);
    expect(component.filteredUserTemplates().length).toBe(1);

    component.onSearchChange('non-matching-query');
    expect(component.filteredPredefinedTemplates().length).toBe(0);
    expect(component.filteredUserTemplates().length).toBe(0);
  });

  it('should switch tabs', () => {
    component.setTab('user');
    expect(component.activeTab()).toBe('user');
    component.setTab('predefined');
    expect(component.activeTab()).toBe('predefined');
  });

  it('should open confirmation dialog and delete user template on confirm', () => {
    const dialogRefSpy = jasmine.createSpyObj({
      afterClosed: of(true),
    });
    mockDialog.open.and.returnValue(dialogRefSpy);
    mockWorkflowService.deleteTemplate.and.returnValue(of(void 0));

    const event = new MouseEvent('click');
    component.deleteUserTemplate(sampleUserTemplate, event);

    expect(mockDialog.open).toHaveBeenCalled();
    expect(mockWorkflowService.deleteTemplate).toHaveBeenCalledWith(
      'user-tmpl-1',
    );
    expect(component.userTemplates().length).toBe(0);
  });
});
