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

import {NodeTypes, StepStatusEnum, WorkflowTemplate} from '../workflow.models';

export const PREDEFINED_WORKFLOW_TEMPLATES: WorkflowTemplate[] = [
  {
    id: 'tmpl-29940221-6bae-41c1-9aee-1ec688313337',
    name: 'Fashion Stylist',
    description: 'Create a weather specific outfit for a specific occasion',
    isPredefined: true,
    steps: [
      {
        stepId: 'user_input',
        status: StepStatusEnum.IDLE,
        outputs: {
          City: {
            type: 'text',
          },
          Occasion_for_outfit: {
            type: 'text',
          },
          'Male,_Female_or_Non-binary': {
            type: 'text',
          },
          'Age,_colors,_style_or_accessories': {
            type: 'text',
          },
        },
        inputs: {},
        settings: {
          definitions: [
            {
              id: 'hw4fsuv64kss6h1bqsasb',
              name: 'City',
              type: 'text',
            },
            {
              id: 'vuwqnrpjlweb3effy0nuf',
              name: 'Male, Female or Non-binary',
              type: 'text',
            },
            {
              id: 'l525z8eu7ioopef3mfig',
              name: 'Occasion for outfit',
              type: 'text',
            },
            {
              id: 'na4wkb7eeqfuzeef02ukf',
              name: 'Age, colors, style or accessories',
              type: 'text',
            },
          ],
        },
        type: NodeTypes.USER_INPUT,
      },
      {
        stepId: 'generate_text_1789154357761',
        status: StepStatusEnum.IDLE,
        outputs: {
          generated_text: {
            type: 'text',
          },
        },
        inputs: {
          prompt:
            'Use the information you get from <city>  to get and return weather conditions',
          input_images: null,
          input_videos: null,
          city: {
            step: 'user_input',
            output: 'User_Text_Input',
          },
        },
        settings: {
          model: 'gemini-3.8-flash',
          temperature: 0.7,
        },
        type: NodeTypes.GENERATE_TEXT,
      },
      {
        stepId: 'generate_text_1789154426585',
        status: StepStatusEnum.IDLE,
        outputs: {
          generated_text: {
            type: 'text',
          },
        },
        inputs: {
          prompt:
            'You\'re a knowledgeable, enthusiastic, and supportive personal wardrobe consultant. You know all the styles, trends and other details appropriate for people of all ages, identities and styles. \nYour goal is to create one outfit that fits the following information. \nWeather conditions: <weather>\nWardrobe requirements: model is a <gender>  occasion: <occasion>, more details are <details>.\nYou must make sure the outfit is appropriate for the weather. For example, you should not recommend a t-shirt as the only top if it\'s 55 degrees F.\nThe outfit should give the title and description as shown below, nothing else:\na brief phrase that identifies this outfit; make it like a memorable one-liner if possible\na short description of the clothing and accessory items that are important for the outfit accompanied by what role they play in making this outfit work for you and your occasion.\n\ngold rule: start you output with "Create an image based on this description below. Be sure to depict the person close up way, being able to see, head to toe, the entire outfit. Any personal details like age, gender race, etc should be reflected in the image:  "',
          input_images: null,
          input_videos: null,
          gender: {
            step: 'user_input',
            output: 'Male,_Female_or_Non-binary',
          },
          details: {
            step: 'user_input',
            output: 'Age,_colors,_style_or_accessories',
          },
          weather: {
            step: 'generate_text_1789154357761',
            output: 'generated_text',
          },
          occasion: {
            step: 'user_input',
            output: 'Occasion_for_outfit',
          },
        },
        settings: {
          model: 'gemini-3.8-flash',
          temperature: 0.7,
        },
        type: NodeTypes.GENERATE_TEXT,
      },
      {
        stepId: 'image_1789155179345',
        status: StepStatusEnum.IDLE,
        outputs: {
          generated_image: {
            type: 'image',
          },
        },
        inputs: {
          prompt: {
            step: 'generate_text_1789154426585',
            output: 'generated_text',
          },
          input_images: null,
          input_image: null,
          model_image: null,
          top_image: null,
          bottom_image: null,
          dress_image: null,
          shoes_image: null,
        },
        settings: {
          mode: 'generate_image',
          model: 'gemini-3.1-flash-image',
          aspect_ratio: '1:1',
          brand_guidelines: false,
          resolution: '1K',
          upscale_factor: 'x2',
          enhance_input_image: false,
          image_preservation_factor: null,
        },
        type: NodeTypes.IMAGE,
      },
    ],
    positions: {
      user_input: {x: 80, y: 120},
      generate_text_1789154357761: {x: 480, y: 120},
      generate_text_1789154426585: {x: 880, y: 120},
      image_1789155179345: {x: 1280, y: 120},
    },
  },
];
